import os
from dotenv import load_dotenv
from openai import OpenAI
import utils

def main():
    # Load environment variables
    load_dotenv()

    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "pt-BR")
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"

    if not API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not found in .env file")

    print("Using DeepSeek API")
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://api.deepseek.com",
        timeout=90.0
    )
    MODEL_NAME = "deepseek-chat"

    utils.run_translation(client, MODEL_NAME, INPUT_DIR, OUTPUT_DIR, TARGET_LANGUAGE, max_workers=100)

if __name__ == "__main__":
    main()
