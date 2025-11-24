import math
import os
import glob
import time
import concurrent.futures
import re
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "pt-BR")
INPUT_DIR = "input"
OUTPUT_DIR = "output"
TARGET_LATENCY = float(os.getenv("TARGET_LATENCY", "8.0"))
# Heuristic for model speed (tokens per second). Adjust if you have measurements.
TOKENS_PER_SEC = float(os.getenv("TOKENS_PER_SEC", "60"))
# Approximate average characters per token for estimation (English ~4)
AVG_CHARS_PER_TOKEN = float(os.getenv("AVG_CHARS_PER_TOKEN", "4"))
# Fixed batch size from env
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "0"))
MAX_TASK_RETRIES = int(os.getenv("MAX_TASK_RETRIES", "3"))
BACKOFF_INITIAL = float(os.getenv("BACKOFF_INITIAL", "1.0"))
BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", "2.0"))

if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in .env file")

# Initialize OpenAI client for DeepSeek
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
    timeout=90.0
)

def read_srt_blocks(file_path):
    """
    Reads an SRT file and splits it into blocks.
    Each block contains the index, timestamp, and text.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Split by double newlines (standard SRT separator)
    # Some files might have slightly different spacing, so we handle that.
    # We assume blocks are separated by at least one empty line.
    blocks = [b.strip() for b in content.split('\n\n') if b.strip()]
    return blocks

def chunk_blocks(blocks, batch_size=None, target_latency=TARGET_LATENCY):
    """
    Chunk blocks dynamically by estimated character/token budget derived from
    `target_latency`, `TOKENS_PER_SEC`, and `AVG_CHARS_PER_TOKEN`.

    If `batch_size` is provided (int), fallback to fixed-size chunking for
    backward compatibility.
    """
    if batch_size and isinstance(batch_size, int):
        for i in range(0, len(blocks), batch_size):
            yield blocks[i:i + batch_size]
        return

    # Estimate allowed characters per chunk from target latency and token rate
    tokens_allowed = max(1.0, target_latency * TOKENS_PER_SEC)
    max_chars = max(200, int(tokens_allowed * AVG_CHARS_PER_TOKEN))

    # Calculate total size and optimal number of chunks to balance load
    total_chars = sum(len(b) for b in blocks)
    if total_chars == 0:
        return

    # Minimum number of chunks needed to respect max_chars (latency limit)
    num_chunks = math.ceil(total_chars / max_chars)
    if num_chunks == 0: num_chunks = 1

    # Target size to distribute load evenly
    target_size = total_chars / num_chunks

    current = []
    current_chars = 0
    for b in blocks:
        b_chars = len(b)
        # If adding this block would exceed the balanced target size, yield current batch.
        # We use target_size (which is <= max_chars) to ensure chunks are roughly equal.
        if current and (current_chars + b_chars) > target_size:
            yield current
            current = []
            current_chars = 0

        current.append(b)
        current_chars += b_chars

        # Safety: if a single block is huge, yield it immediately
        if current_chars >= max_chars * 2:
            yield current
            current = []
            current_chars = 0

    if current:
        yield current

def translate_batch(batch_blocks, target_lang, batch_index, file_path=None):
    """
    Sends a batch of SRT blocks to the LLM for translation.
    """
    input_text = "\n\n".join(batch_blocks)
    # char_count = len(input_text)
    char_count = len(input_text)
    est_tokens = int(char_count / AVG_CHARS_PER_TOKEN)
    file_label = os.path.basename(file_path) if file_path else "<unknown>"
    tqdm.write(f"[BATCH] File:{file_label} Batch:{batch_index} Blocks:{len(batch_blocks)} Chars:{char_count} EstTokens:{est_tokens}")

    system_prompt = (
        f"You are a professional subtitle translator. Translate the following SRT subtitles to {target_lang}. "
        "Strictly adhere to these rules:\n"
        "1. Maintain the exact SRT format (Index, Timestamp, Text).\n"
        "2. Do NOT translate the speaker names (e.g., 'Gordon:', 'Cameron:') if they appear before the dialogue.\n"
        "3. Do NOT translate character names like 'Big Bird', 'Fresh Prince', etc.\n"
        "4. Preserve ALL HTML tags (<i>, <b>, etc.) exactly where they are.\n"
        "5. Do NOT change the timestamps or indices.\n"
        "6. Output ONLY the translated SRT blocks, separated by double newlines. No introductory text."
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                temperature=0.3,
                stream=False
            )
            duration = time.time() - start_time
            tqdm.write(f"[BATCH] File:{file_label} Batch:{batch_index} Finished in {duration:.2f}s")
            return response.choices[0].message.content.strip()
        except Exception as e:
            tqdm.write(f"Error translating batch {batch_index} (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2) # Wait a bit before retrying

def save_translated_file(original_path, translated_parts, target_lang):
    """
    Saves the translated parts to a file.
    """
    # Filter out None results (failed batches)
    valid_parts = [p for p in translated_parts if p is not None]

    if not valid_parts:
        print(f"No content translated for {os.path.basename(original_path)}")
        return

    final_content = "\n\n".join(valid_parts)
    filename = os.path.basename(original_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}.{target_lang}{ext}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
    print(f"\nSaved: {output_filename}")


def validate_translated_chunk(original_blocks, translated_text):
    """
    Validate that the translated chunk preserves indices and timestamps.
    Returns True if valid, False otherwise.
    """
    try:
        translated_blocks = [b.strip() for b in translated_text.split('\n\n') if b.strip()]
        if len(translated_blocks) != len(original_blocks):
            return False

        timestamp_re = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$")
        for orig, tr in zip(original_blocks, translated_blocks):
            orig_lines = orig.splitlines()
            tr_lines = tr.splitlines()
            if len(orig_lines) < 2 or len(tr_lines) < 2:
                return False
            # Check index (first line) is identical
            if orig_lines[0].strip() != tr_lines[0].strip():
                return False
            # Check timestamp (second line) matches original exactly
            if orig_lines[1].strip() != tr_lines[1].strip():
                return False
            # Also validate timestamp format
            if not timestamp_re.match(tr_lines[1].strip()):
                return False
        return True
    except Exception:
        return False

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all SRT files in input directory
    srt_files = glob.glob(os.path.join(INPUT_DIR, "*.srt"))

    if not srt_files:
        print(f"No SRT files found in {INPUT_DIR}")
        return

    print(f"Found {len(srt_files)} SRT files to process.")
    print(f"Target Language: {TARGET_LANGUAGE}")

    # Prepare all tasks across all files
    all_tasks = []
    file_progress = {}

    print("Preparing tasks...")
    for file_path in srt_files:
        # Check if output file already exists
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}.{TARGET_LANGUAGE}{ext}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        if os.path.exists(output_path):
            print(f"Skipping existing translation: {output_filename}")
            continue

        blocks = read_srt_blocks(file_path)
        if not blocks:
            print(f"Skipping empty file: {os.path.basename(file_path)}")
            continue

        # Use fixed batch size if configured, otherwise dynamic
        b_size = BATCH_SIZE if BATCH_SIZE > 0 else None
        batches = list(chunk_blocks(blocks, batch_size=b_size))

        file_progress[file_path] = {
            'parts': [None] * len(batches),
            'remaining': len(batches)
        }

        for i, batch in enumerate(batches):
            all_tasks.append((file_path, i, batch))

    print(f"Total tasks: {len(all_tasks)} across {len(srt_files)} files.")

    # Use a high number of workers to maximize throughput across all files
    max_workers = 50
    print(f"Starting global translation with {max_workers} parallel workers...")

    # Map task key -> original batch blocks for validation/retries
    batches_map = {(fp, idx): batch for fp, idx, batch in all_tasks}
    # Track attempts per task
    attempts = { (fp, idx): 0 for fp, idx, _ in all_tasks }

    total_tasks = len(all_tasks)
    progress = tqdm(total=total_tasks, desc="Translating all")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all initial tasks
        future_set = set()
        future_to_info = {}
        for file_path, idx, batch in all_tasks:
            fut = executor.submit(translate_batch, batch, TARGET_LANGUAGE, idx, file_path)
            future_set.add(fut)
            future_to_info[fut] = (file_path, idx)

        # Process futures as they complete; allow resubmission on failure/validation
        while future_set:
            done, _ = concurrent.futures.wait(future_set, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                future_set.remove(fut)
                file_path, idx = future_to_info.pop(fut)
                batch_blocks = batches_map[(file_path, idx)]
                try:
                    result = fut.result()
                except Exception as e:
                    result = None
                    tqdm.write(f"Task exception for {os.path.basename(file_path)} batch {idx}: {e}")

                valid = False
                if result is not None:
                    # Validate translated chunk preserves indices/timestamps
                    if validate_translated_chunk(batch_blocks, result):
                        valid = True
                    else:
                        tqdm.write(f"Validation failed for {os.path.basename(file_path)} batch {idx}")

                if valid:
                    file_progress[file_path]['parts'][idx] = result
                    file_progress[file_path]['remaining'] -= 1
                    progress.update(1)
                    # If file complete, save it
                    if file_progress[file_path]['remaining'] == 0:
                        save_translated_file(file_path, file_progress[file_path]['parts'], TARGET_LANGUAGE)
                else:
                    # Need to retry or fail permanently
                    attempts[(file_path, idx)] += 1
                    if attempts[(file_path, idx)] <= MAX_TASK_RETRIES:
                        backoff = BACKOFF_INITIAL * (BACKOFF_FACTOR ** (attempts[(file_path, idx)] - 1))
                        tqdm.write(f"Requeueing {os.path.basename(file_path)} batch {idx} (attempt {attempts[(file_path, idx)]}) after {backoff:.1f}s")
                        time.sleep(backoff)
                        fut2 = executor.submit(translate_batch, batch_blocks, TARGET_LANGUAGE, idx, file_path)
                        future_set.add(fut2)
                        future_to_info[fut2] = (file_path, idx)
                    else:
                        tqdm.write(f"Batch {idx} for {os.path.basename(file_path)} failed permanently after {attempts[(file_path, idx)]} attempts")
                        file_progress[file_path]['parts'][idx] = None
                        file_progress[file_path]['remaining'] -= 1
                        progress.update(1)
                        if file_progress[file_path]['remaining'] == 0:
                            save_translated_file(file_path, file_progress[file_path]['parts'], TARGET_LANGUAGE)

        progress.close()

if __name__ == "__main__":
    print("Starting script...")
    main()
