import os, re, json, pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types

# ------------ paths & parameters ------------
INPUT_CSV  = "self_annotation/data_to_annotate.csv"
OUTPUT_JSONL = "test/gemini_predictions.jsonl"
MODEL      = "gemini-2.5-flash-preview-05-20"
# ---------- paths & parameters ----------
INPUT_CSV  = "self_annotation/data_to_annotate.csv"
OUTPUT_CSV = "test/gemini_predictions.csv"
MODEL      = "gemini-2.5-flash-preview-05-20"
# --- Prompt (no changes needed) ---
PROMPT_TEMPLATE = """Below is an extract from a web page. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:
- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.
- Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.
- Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students.
- Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren't too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.
- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.

The extract:
{text_to_evaluate}

After examining the extract:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Educational score: <total points>"
"""

# ------------ helper ------------
def score_extract(text: str, client, cfg) -> int:
    if not isinstance(text, str) or not text.strip():
        return -1
    try:
        prompt = PROMPT_TEMPLATE.format(text_to_evaluate=text)
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=cfg,
        )
        m = re.search(r"Educational score:\s*(\d+)", resp.text, re.I)
        return int(m.group(1)) if m else -1
    except Exception as e:
        print("API error:", e)
        return -1

# ------------ main ------------
if __name__ == "__main__":
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set.")

    client = genai.Client(api_key=key)
    gen_cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="text/plain",
    )

    print(f"Loading {INPUT_CSV} …")
    df = pd.read_csv(INPUT_CSV)

    # open output file once for streaming write
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        print(f"Scoring {len(df)} extracts and writing to {OUTPUT_JSONL} …")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            result = {
                "text": row["text"],
                "gemini_prediction": score_extract(row["text"], client, gen_cfg),
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✓ Predictions saved to {OUTPUT_JSONL}")