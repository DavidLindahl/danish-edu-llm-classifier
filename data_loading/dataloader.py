import pandas as pd
from datatrove.pipeline.readers import ParquetReader
from tqdm import tqdm
import os

def dataloader(samples=100, to_csv=False, csv_filename="output.csv"):
    """"""
# Initialize the ParquetReader
    data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb-2/data/dan_Latn/train", limit=samples , shuffle_files=True)

# List to store extracted data
    data_list = []

# Loop through each document and extract relevant fields
    for document in tqdm(data_reader()):  
        data_list.append({
            "text": document.text,
            "id": document.id,
            "url": document.metadata.get("url", ""),
            "language": document.metadata.get("language", ""),
            "language_score": document.metadata.get("language_score", 0)
        })

    # Convert list to DataFrame
    df = pd.DataFrame(data_list)
    print(df.head())
    # Save to CSV file
    if to_csv:
        os.makedirs("data", exist_ok=True)
        csv_path = os.path.join("data", csv_filename)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"CSV file saved as {csv_path}")

    return df

if __name__ == "__main__":
    df = dataloader(samples=100, to_csv=True, csv_filename="training_data.csv")
    print("Data loading complete.")
    print(df.head())