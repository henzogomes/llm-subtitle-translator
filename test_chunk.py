import os
import time
import glob
import argparse
from dotenv import load_dotenv
from openai import OpenAI
import utils

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Test translation on a single chunk.")
    parser.add_argument("--provider", choices=["ollama", "deepseek"], default="ollama", help="LLM provider to use")
    args = parser.parse_args()

    TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "pt-BR")
    INPUT_DIR = "input"

    if args.provider == "ollama":
        OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        MODEL_NAME = os.getenv("OLLAMA_MODEL", "deepseek-r1")
        print(f"Testing with Ollama (Model: {MODEL_NAME})")
        print(f"Base URL: {OLLAMA_BASE_URL}")
        client = OpenAI(
            api_key="ollama",
            base_url=OLLAMA_BASE_URL,
            timeout=600.0
        )
    elif args.provider == "deepseek":
        API_KEY = os.getenv("DEEPSEEK_API_KEY")
        if not API_KEY:
            raise ValueError("DEEPSEEK_API_KEY not found in .env file")
        MODEL_NAME = "deepseek-chat"
        print("Testing with DeepSeek API")
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://api.deepseek.com",
            timeout=90.0
        )

    srt_files = glob.glob(os.path.join(INPUT_DIR, "*.srt"))
    if not srt_files:
        print("No SRT files found.")
        return

    file_path = srt_files[0]
    print(f"Reading file: {file_path}")

    blocks = utils.read_srt_blocks(file_path)
    if not blocks:
        print("File is empty.")
        return

    # Get just the first batch
    batches = list(utils.chunk_blocks(blocks, batch_size=None))
    if not batches:
        print("No batches created.")
        return

    first_batch = batches[0]
    print(f"First batch has {len(first_batch)} blocks.")

    # Use the translate_batch function from utils to test the exact logic used in production
    print("\n--- Sending Request via utils.translate_batch ---")
    translated_text = utils.translate_batch(client, MODEL_NAME, first_batch, TARGET_LANGUAGE, 0, file_path)

    if translated_text:
        print("\n--- Response Content ---")
        print(translated_text)
        print("\n--- End of Response ---")

        # Validate
        print("\n--- Validation Check ---")
        is_valid = utils.validate_translated_chunk(first_batch, translated_text)
        print(f"Validation passed: {is_valid}")
    else:
        print("Translation failed (returned None).")

if __name__ == "__main__":
    main()
