import math
import os
import glob
import time
import concurrent.futures
import re
from tqdm import tqdm

# Configuration Defaults
TARGET_LATENCY = float(os.getenv("TARGET_LATENCY", "8.0"))
TOKENS_PER_SEC = float(os.getenv("TOKENS_PER_SEC", "60"))
AVG_CHARS_PER_TOKEN = float(os.getenv("AVG_CHARS_PER_TOKEN", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "0"))
MAX_TASK_RETRIES = int(os.getenv("MAX_TASK_RETRIES", "3"))
BACKOFF_INITIAL = float(os.getenv("BACKOFF_INITIAL", "1.0"))
BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", "2.0"))

def read_srt_blocks(file_path):
    """
    Reads an SRT file and splits it into blocks.
    Each block contains the index, timestamp, and text.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    blocks = [b.strip() for b in content.split('\n\n') if b.strip()]
    return blocks

def chunk_blocks(blocks, batch_size=None, target_latency=TARGET_LATENCY):
    """
    Chunk blocks dynamically by estimated character/token budget.
    """
    if batch_size and isinstance(batch_size, int) and batch_size > 0:
        for i in range(0, len(blocks), batch_size):
            yield blocks[i:i + batch_size]
        return

    tokens_allowed = max(1.0, target_latency * TOKENS_PER_SEC)
    max_chars = max(200, int(tokens_allowed * AVG_CHARS_PER_TOKEN))

    total_chars = sum(len(b) for b in blocks)
    if total_chars == 0:
        return

    num_chunks = math.ceil(total_chars / max_chars)
    if num_chunks == 0: num_chunks = 1

    target_size = total_chars / num_chunks

    current = []
    current_chars = 0
    for b in blocks:
        b_chars = len(b)
        if current and (current_chars + b_chars) > target_size:
            yield current
            current = []
            current_chars = 0

        current.append(b)
        current_chars += b_chars

        if current_chars >= max_chars * 2:
            yield current
            current = []
            current_chars = 0

    if current:
        yield current

def translate_batch(client, model_name, batch_blocks, target_lang, batch_index, file_path=None):
    """
    Sends a batch of SRT blocks to the LLM for translation.
    """
    input_text = "\n\n".join(batch_blocks)
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

    for attempt in range(MAX_TASK_RETRIES):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                temperature=0.3,
                stream=False
            )
            duration = time.time() - start_time
            tqdm.write(f"[BATCH] File:{file_label} Batch:{batch_index} Finished in {duration:.2f}s")

            content = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content.rsplit("\n", 1)[0]
            return content.strip()
        except Exception as e:
            tqdm.write(f"Error translating batch {batch_index} (Attempt {attempt + 1}/{MAX_TASK_RETRIES}): {e}")
            if attempt == MAX_TASK_RETRIES - 1:
                return None
            time.sleep(2)

def save_translated_file(original_path, translated_parts, target_lang, output_dir):
    """
    Saves the translated parts to a file.
    """
    valid_parts = [p for p in translated_parts if p is not None]

    if not valid_parts:
        print(f"No content translated for {os.path.basename(original_path)}")
        return

    final_content = "\n\n".join(valid_parts)
    filename = os.path.basename(original_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}.{target_lang}{ext}"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
    print(f"\nSaved: {output_filename}")

def validate_translated_chunk(original_blocks, translated_text):
    """
    Validate that the translated chunk preserves indices and timestamps.
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
            if orig_lines[0].strip() != tr_lines[0].strip():
                return False
            if orig_lines[1].strip() != tr_lines[1].strip():
                return False
            if not timestamp_re.match(tr_lines[1].strip()):
                return False
        return True
    except Exception:
        return False

def run_translation(client, model_name, input_dir, output_dir, target_lang, max_workers=10):
    """
    Main orchestration function.
    """
    os.makedirs(output_dir, exist_ok=True)
    srt_files = glob.glob(os.path.join(input_dir, "*.srt"))

    if not srt_files:
        print(f"No SRT files found in {input_dir}")
        return

    print(f"Found {len(srt_files)} SRT files to process.")
    print(f"Target Language: {target_lang}")
    print(f"Model: {model_name}")

    all_tasks = []
    file_progress = {}

    print("Preparing tasks...")
    for file_path in srt_files:
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}.{target_lang}{ext}"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            print(f"Skipping existing translation: {output_filename}")
            continue

        blocks = read_srt_blocks(file_path)
        if not blocks:
            print(f"Skipping empty file: {os.path.basename(file_path)}")
            continue

        b_size = BATCH_SIZE if BATCH_SIZE > 0 else None
        batches = list(chunk_blocks(blocks, batch_size=b_size))

        file_progress[file_path] = {
            'parts': [None] * len(batches),
            'remaining': len(batches)
        }

        for i, batch in enumerate(batches):
            all_tasks.append((file_path, i, batch))

    print(f"Total tasks: {len(all_tasks)} across {len(srt_files)} files.")

    print(f"Starting global translation with {max_workers} parallel workers...")

    batches_map = {(fp, idx): batch for fp, idx, batch in all_tasks}
    attempts = { (fp, idx): 0 for fp, idx, _ in all_tasks }

    total_tasks = len(all_tasks)
    if total_tasks == 0:
        print("Nothing to translate.")
        return

    progress = tqdm(total=total_tasks, desc="Translating all")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_set = set()
        future_to_info = {}
        for file_path, idx, batch in all_tasks:
            fut = executor.submit(translate_batch, client, model_name, batch, target_lang, idx, file_path)
            future_set.add(fut)
            future_to_info[fut] = (file_path, idx)

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
                    if validate_translated_chunk(batch_blocks, result):
                        valid = True
                    else:
                        tqdm.write(f"Validation failed for {os.path.basename(file_path)} batch {idx}")

                if valid:
                    file_progress[file_path]['parts'][idx] = result
                    file_progress[file_path]['remaining'] -= 1
                    progress.update(1)
                    if file_progress[file_path]['remaining'] == 0:
                        save_translated_file(file_path, file_progress[file_path]['parts'], target_lang, output_dir)
                else:
                    attempts[(file_path, idx)] += 1
                    if attempts[(file_path, idx)] <= MAX_TASK_RETRIES:
                        backoff = BACKOFF_INITIAL * (BACKOFF_FACTOR ** (attempts[(file_path, idx)] - 1))
                        tqdm.write(f"Requeueing {os.path.basename(file_path)} batch {idx} (attempt {attempts[(file_path, idx)]}) after {backoff:.1f}s")
                        time.sleep(backoff)
                        fut2 = executor.submit(translate_batch, client, model_name, batch_blocks, target_lang, idx, file_path)
                        future_set.add(fut2)
                        future_to_info[fut2] = (file_path, idx)
                    else:
                        tqdm.write(f"Batch {idx} for {os.path.basename(file_path)} failed permanently after {attempts[(file_path, idx)]} attempts")
                        file_progress[file_path]['parts'][idx] = None
                        file_progress[file_path]['remaining'] -= 1
                        progress.update(1)
                        if file_progress[file_path]['remaining'] == 0:
                            save_translated_file(file_path, file_progress[file_path]['parts'], target_lang, output_dir)

    progress.close()
