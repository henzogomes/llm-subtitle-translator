import os
import glob
import time
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "pt-BR")
INPUT_DIR = "input"
OUTPUT_DIR = "output"

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

def chunk_blocks(blocks, batch_size=10):
    """
    Yields successive batches of blocks.
    """
    for i in range(0, len(blocks), batch_size):
        yield blocks[i:i + batch_size]

def translate_batch(batch_blocks, target_lang, batch_index):
    """
    Sends a batch of SRT blocks to the LLM for translation.
    """
    input_text = "\n\n".join(batch_blocks)
    # char_count = len(input_text)
    # print(f"\n[DEBUG] Batch {batch_index}: Sending {len(batch_blocks)} blocks ({char_count} chars)...")

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
            # start_time = time.time()
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                temperature=0.3,
                stream=False
            )
            # duration = time.time() - start_time
            # print(f"[DEBUG] Batch {batch_index}: Finished in {duration:.2f}s")
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error translating batch {batch_index} (Attempt {attempt + 1}/{max_retries}): {e}")
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

        batches = list(chunk_blocks(blocks))
        file_progress[file_path] = {
            'parts': [None] * len(batches),
            'remaining': len(batches)
        }

        for i, batch in enumerate(batches):
            all_tasks.append((file_path, i, batch))

    print(f"Total tasks: {len(all_tasks)} across {len(srt_files)} files.")

    # Use a high number of workers to maximize throughput across all files
    max_workers = 100
    print(f"Starting global translation with {max_workers} parallel workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_info = {}
        for file_path, idx, batch in all_tasks:
            future = executor.submit(translate_batch, batch, TARGET_LANGUAGE, idx)
            future_to_info[future] = (file_path, idx)

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_info), total=len(all_tasks), desc="Translating all"):
            file_path, idx = future_to_info[future]
            try:
                result = future.result()
                file_progress[file_path]['parts'][idx] = result
            except Exception as e:
                print(f"Task failed for {os.path.basename(file_path)} batch {idx}: {e}")

            # Decrement remaining count for this file
            file_progress[file_path]['remaining'] -= 1

            # Check if this specific file is complete
            if file_progress[file_path]['remaining'] == 0:
                save_translated_file(file_path, file_progress[file_path]['parts'], TARGET_LANGUAGE)

if __name__ == "__main__":
    print("Starting script...")
    main()
