"""
Script to evaluate educational value of text content using Google Gemini Flash 2.5 API.
Reads from training_data_combined.csv and evaluates each text sample.
"""

import pandas as pd
import google.generativeai as genai
import os
import time
from tqdm import tqdm
import json

def load_api_key():
    """Load API key from file."""
    api_key_path = os.path.join("data", "gemini_api_key.txt")
    try:
        with open(api_key_path, 'r') as f:
            api_key = f.read().strip()
        return api_key
    except FileNotFoundError:
        print(f"Error: API key file not found at {api_key_path}")
        print("Please create the file and add your Gemini API key.")
        return None
    except Exception as e:
        print(f"Error reading API key file: {e}")
        return None

# Load and configure API key
api_key = load_api_key()
if api_key:
    genai.configure(api_key=api_key)
    print("✅ API key loaded successfully")
else:
    print("❌ Failed to load API key")
    exit(1)

def get_educational_score(text_content):
    """
    Evaluate educational value of text content using Gemini Flash 2.5 API.
    
    Args:
        text_content (str): The text to evaluate
        
    Returns:
        dict: Dictionary containing score and justification
    """
    
    prompt = f"""Below is an extract from a web page. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:
    •	Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.
    •	Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.
    •	Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students.
    •	Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren't too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.
    •	Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.

The extract: {text_content}

After examining the extract:
    •	Briefly justify your total score, up to 100 words.
    •	Conclude with the score using the format: "Educational score: X"
"""

    try:
        # Initialize the model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=200,
                temperature=0.3,
            )
        )
        
        response_text = response.text.strip()
        
        # Extract score from response
        score = None
        justification = response_text
        
        if "Educational score:" in response_text:
            score_line = response_text.split("Educational score:")[-1].strip()
            try:
                score = int(score_line.split()[0])
            except:
                score = None
        
        return {
            "score": score,
            "justification": justification,
            "full_response": response_text
        }
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return {
            "score": None,
            "justification": f"Error: {e}",
            "full_response": None
        }

def main():
    """Main function to evaluate educational content from CSV file."""
    
    # Path to the combined CSV file
    csv_path = os.path.join("data", "clean_data.csv")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found!")
        print("Current directory:", os.getcwd())
        print("Available files in data folder:")
        if os.path.exists("data"):
            print(os.listdir("data"))
        else:
            print("Data folder doesn't exist!")
        return
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        print(f"Loaded {len(df)} samples from {csv_path}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Initialize list to store all results
    all_results = []
    
    # Ask user how many samples to evaluate
    max_samples = len(df)
    num_samples = input(f"How many samples to evaluate? (max {max_samples}, press Enter for all): ")
    
    if num_samples.strip() == "":
        num_samples = max_samples
    else:
        try:
            num_samples = min(int(num_samples), max_samples)
        except:
            num_samples = max_samples
    
    print(f"Evaluating {num_samples} samples...")
    
    # Evaluate each text sample
    for i in tqdm(range(num_samples), desc="Evaluating samples"):
        row = df.iloc[i]
        text_content = str(row['text'])
        
        # Truncate very long texts to avoid API limits
        if len(text_content) > 4000:
            text_content = text_content[:4000] + "..."
        
        result = get_educational_score(text_content)
        
        # Create record with original data plus evaluation results
        record = {
            "id": row['id'],
            "url": row['url'],
            "language": row['language'],
            "language_score": row['language_score'],
            "text": row['text'],
            "educational_score": result['score'],
            "score_justification": result['justification'],
            "full_evaluation": result['full_response'],
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        all_results.append(record)
        
        # Add small delay to respect API rate limits
        time.sleep(0.3)
        
        # Print progress every 10 samples
        if (i + 1) % 10 == 0:
            valid_scores = [r['educational_score'] for r in all_results if r['educational_score'] is not None]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                print(f"Progress: {i+1}/{num_samples} - Average score so far: {avg_score:.2f}")
    
    # Save results as JSON
    json_output_path = os.path.join("data", "evaluated_training_data.json")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Also save a summary CSV with just the scores
    summary_data = []
    for record in all_results:
        summary_data.append({
            "id": record["id"],
            "url": record["url"],
            "educational_score": record["educational_score"],
            "text_length": len(record["text"]) if record["text"] else 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join("data", "educational_scores_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")
    
    # Print summary
    valid_scores = [r['educational_score'] for r in all_results if r['educational_score'] is not None]
    if valid_scores:
        print(f"\n=== Evaluation Complete ===")
        print(f"Samples evaluated: {len(all_results)}")
        print(f"Valid scores: {len(valid_scores)}")
        print(f"Average educational score: {sum(valid_scores) / len(valid_scores):.2f}")
        print(f"Score distribution:")
        for score in range(1, 6):
            count = valid_scores.count(score)
            print(f"  Score {score}: {count} samples ({count/len(valid_scores)*100:.1f}%)")
        print(f"\nResults saved to:")
        print(f"  📁 Full data (JSON): {json_output_path}")
        print(f"  📊 Summary (CSV): {summary_csv_path}")
    else:
        print("No valid scores obtained. Check your API key and connection.")

if __name__ == "__main__":
    main()