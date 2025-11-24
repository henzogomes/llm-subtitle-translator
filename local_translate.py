import os
from dotenv import load_dotenv
from openai import OpenAI
import utils

def main():
    # Load environment variables
    load_dotenv()

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1")
    TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "pt-BR")
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"

    print(f"Using Ollama (Model: {OLLAMA_MODEL})")
    client = OpenAI(
        api_key="ollama",  # Ollama doesn't require a real key, but the client needs one
        base_url=OLLAMA_BASE_URL,
        timeout=600.0  # Increased timeout for local CPU inference
    )

    # For local LLMs, we use 1 worker to avoid queuing and timeouts,
    # as the model typically consumes all available compute resources.
    utils.run_translation(client, OLLAMA_MODEL, INPUT_DIR, OUTPUT_DIR, TARGET_LANGUAGE, max_workers=1)

if __name__ == "__main__":
    main()
