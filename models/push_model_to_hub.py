from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse

def push_model(local_path, repo_id):
    """
    Loads a model and tokenizer from a local path and pushes them to the Hub.
    
    Args:
        local_path (str): The local directory where the model is saved.
        repo_id (str): The desired repository ID on the Hub (e.g., "YourUsername/YourModelName").
    """
    print(f"Loading model and tokenizer from '{local_path}'...")
    model = AutoModelForSequenceClassification.from_pretrained(local_path)
    tokenizer = AutoTokenizer.from_pretrained(local_path)

    print(f"Pushing model to '{repo_id}'...")
    # You can add private=True to make the repo private initially
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    print("--- Push complete! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push a local model to the Hugging Face Hub.")
    parser.add_argument("--local_path", type=str, required=True, help="Local path to the model directory.")
    parser.add_argument("--repo_id", type=str, required=True, help="Hub repository ID (e.g., YourUsername/YourModelName).")
    
    args = parser.parse_args()
    push_model(args.local_path, args.repo_id)