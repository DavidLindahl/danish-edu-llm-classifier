import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import os
import torch_xla
import torch_xla.core.xla_model as xm

# --- Configuration ---
MODEL_NAME = "Helsinki-NLP/opus-mt-en-da"
INPUT_CSV_PATH = "data/english_fineweb_merged_data_5k_danish.csv"
OUTPUT_CSV_PATH = "data/danish_train.csv"
TEXT_COLUMN_NAME = "text"

# Adjust batch size based on your GPU's VRAM. 
# A larger batch size is faster but uses more memory.
# Start with 32 if you have a modern GPU, or 8-16 for older ones.
BATCH_SIZE = 128 

def translate_texts_in_batches(texts_to_translate, model, tokenizer, device, batch_size):
    """
    Translates a list of texts in batches to optimize for speed and memory.

    Args:
        texts_to_translate (list): A list of strings to be translated.
        model: The pre-loaded translation model.
        tokenizer: The pre-loaded tokenizer.
        device (torch.device): The device (CPU or CUDA) to run the model on.
        batch_size (int): The number of texts to process in a single batch.

    Returns:
        list: A list of translated strings.
    """
    danish_translations = []
    print(f"Starting translation of {len(texts_to_translate)} texts in batches of {batch_size}...")

    # Wrap the loop with tqdm for a progress bar
    for i in tqdm(range(0, len(texts_to_translate), batch_size), desc="Translating Batches"):
        # 1. Create a batch of texts
        batch = texts_to_translate[i : i + batch_size]
        
        # 2. Tokenize the batch. `padding=True` makes all sequences in the batch the same length.
        # `truncation=True` cuts off texts longer than the model can handle.
        # `return_tensors="pt"` returns PyTorch tensors.
        tokenized_batch = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)

        # 3. Generate the translation using the model
        translated_tokens = model.generate(**tokenized_batch)

        # 4. Decode the generated tokens back into text
        # `skip_special_tokens=True` removes tokens like <s> and </s>
        translated_batch = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        
        # 5. Add the translated batch to our results list
        danish_translations.extend(translated_batch)

    return danish_translations

def main():
    """
    Main function to orchestrate the CSV translation process.
    """
    # --- 1. Setup Environment for TPU ---


    # This is the magic command that finds and initializes the Colab TPU
    device = xm.xla_device()
    print(f"Successfully connected to TPU device: {device}")

    # --- 2. Load Model and Tokenizer ---
    print(f"Loading model and tokenizer: {MODEL_NAME}")
    try:
        tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)
    except OSError:
        print(f"Error: Model '{MODEL_NAME}' not found. Please check the model name and your internet connection.")
        return
    print("Model loaded successfully.")

    # --- 3. Load and Prepare Data ---
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file not found at '{INPUT_CSV_PATH}'")
        return
        
    print(f"Reading input CSV: {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)

    if TEXT_COLUMN_NAME not in df.columns:
        print(f"Error: Text column '{TEXT_COLUMN_NAME}' not found in the CSV.")
        print(f"Available columns are: {list(df.columns)}")
        return
        
    # Ensure all entries in the text column are strings to avoid errors
    texts_to_translate = df[TEXT_COLUMN_NAME].astype(str).tolist()

    # --- 4. Perform Translation ---
    translated_texts = translate_texts_in_batches(
        texts_to_translate, model, tokenizer, device, BATCH_SIZE
    )

    # --- 5. Update DataFrame and Save Output ---
    print("Updating DataFrame with translations...")
    df[TEXT_COLUMN_NAME] = translated_texts

    # Ensure the output directory exists
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving translated data to: {OUTPUT_CSV_PATH}")
    # Use 'utf-8-sig' to ensure Danish characters are handled correctly, especially in Excel
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    # --- Add this line before the final print statements ---
    # This waits for all asynchronous operations on the TPU to complete.
    xm.wait_for_devices()

    print("\nTranslation complete!")
    print(f"Output file created at: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()