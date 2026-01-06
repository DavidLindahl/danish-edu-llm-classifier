"""
Streamlit Annotation Tool for Educational Content

This script launches a web-based tool for annotating the educational
quality of text samples. It reads from a source CSV, allows a user to
assign a label, and saves the annotations to a JSON file.

Usage:
------
To run this application, navigate to the project's root directory
in your terminal and execute the following command:

streamlit run src/annotation/calculate_agreement.py
"""

import os

import pandas as pd
import streamlit as st  # pip install streamlit pandas

# File paths
folder = ""  # e.g., "intro_code/"
RAW_DATA_FILE = "src/annotation/data_to_annotate.csv"
ANNOTATIONS_FILE = "src/annotation/annotations.json"


# Load raw data (CSV stays the same)
@st.cache_data
def load_raw_data():
    return pd.read_csv(RAW_DATA_FILE)


# Load or create annotation file (JSON now)
def load_annotations():
    if os.path.exists(ANNOTATIONS_FILE):
        try:
            return pd.read_json(ANNOTATIONS_FILE)
        except ValueError:
            # If the JSON file is empty or corrupted, return empty DataFrame
            return pd.DataFrame(columns=["text", "original_label", "our_label"])
    else:
        return pd.DataFrame(columns=["text", "original_label", "our_label"])


# Save new annotation (JSON now)
def save_annotation(new_row):
    df = load_annotations()
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_json(ANNOTATIONS_FILE, orient="records", indent=2)


# Main Streamlit app
def main():
    st.title("Educational Quality Annotation Tool")

    raw_data = load_raw_data()
    annotations = load_annotations()

    # how many annotations
    total_samples = len(raw_data)
    annotated_count = len(annotations)
    st.markdown(f"**Progress:** {annotated_count} / {total_samples} annotations done")

    # Filter out already annotated samples
    annotated_texts = set(annotations["text"])
    remaining_data = raw_data[~raw_data["text"].isin(annotated_texts)]

    if remaining_data.empty:
        st.success("✅ All data has been annotated!")
        return

    # Select a row to annotate
    sample = remaining_data.iloc[0]
    st.write("### Text to annotate:")
    st.write(sample["text"])

    # Show original label for comparison
    original_label = sample.get("educational_value_labels", "N/A")
    # st.markdown(f"**Original Label:** `{original_label}`")

    # Annotation options
    label = st.radio(
        "Select educational quality level:",
        ["None", "Minimal", "Basic", "Good", "Excellent", "❗ Problematic Content ❗"],
        key="label_radio",
    )

    if st.button("Submit annotation"):
        save_annotation(
            {
                "id": sample["id"],
                "text": sample["text"],
                "original_label": original_label,
                "our_label": label,
            }
        )
        st.success("✅ Annotation saved. Reloading next example...")
        st.rerun()  # updated method for rerun


if __name__ == "__main__":
    main()
