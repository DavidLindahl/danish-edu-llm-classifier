import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import os

# NOTE: Removed torch_xla imports as they are for TPUs only

# --- Configuration ---
MODEL_NAME = "Helsinki-NLP/opus-mt-en-da"
INPUT_CSV_PATH = "data/danish_pretranslate.csv"
OUTPUT_CSV_PATH = "data/processed/danish_train.csv"
TEXT_COLUMN_NAME = "tran_text"
BATCH_SIZE = 32 

def translate_texts_in_batches(texts_to_translate, model, tokenizer, device, batch_size):
    """
    Translates a list of texts in batches to optimize for speed and memory.
    """
    danish_translations = []
    print(f"Starting translation of {len(texts_to_translate)} texts in batches of {batch_size}...")

    for i in tqdm(range(0, len(texts_to_translate), batch_size), desc="Translating Batches"):
        batch = texts_to_translate[i : i + batch_size]
        
        # Move the tokenized batch to the specified device (MPS or CPU)
        tokenized_batch = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)

        translated_tokens = model.generate(**tokenized_batch)
        translated_batch = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        danish_translations.extend(translated_batch)

    return danish_translations

def main():
    """
    Main function to orchestrate the CSV translation process on a Mac.
    """
    # --- 1. Setup Environment for Mac ---
    # This logic correctly finds your M2 GPU (MPS) or falls back to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: Apple M2 GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU (Apple's MPS backend not available)")

    # --- 2. Load Model and Tokenizer ---
    print(f"Loading model and tokenizer: {MODEL_NAME}")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    # Move the model to the Mac's GPU
    model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)
    print("Model loaded successfully.")

    # --- 3. Load and Prepare Data ---
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file not found at '{INPUT_CSV_PATH}'")
        return
        
    print(f"Reading input CSV: {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)
    texts_to_translate = df[TEXT_COLUMN_NAME].astype(str).tolist()

    # --- 4. Perform Translation ---
    translated_texts = translate_texts_in_batches(
        texts_to_translate, model, tokenizer, device, BATCH_SIZE
    )

    # --- 5. Update DataFrame and Save Output ---
    print("Updating DataFrame with translations...")
    df[TEXT_COLUMN_NAME] = translated_texts
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    
    # NOTE: Removed xm.wait_for_devices() as it is for TPUs only

    print("\nTranslation complete!")
    print(f"Output file created at: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()