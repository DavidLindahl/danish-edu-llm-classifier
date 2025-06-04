# danish-edu-llm-classifier
Analysis and enhancement of the FineWeb-C dataset for multilingual educational content classification using LLMs, with a focus on annotation quality and inter-annotator agreement.


## Data Folder

This folder contains all datasets used for training and evaluating the educational content classifier models. Below is a description of each file and how the data was processed and labeled.

---

### Danish Data

- **danish_unfiltered_data.csv**  
  The raw Danish dataset, collected from the FineWeb dataset using the script `get_danish_data.py`. This file contains all samples, including those that may be problematic or irrelevant for educational purposes.

- **danish_filtered_data.csv**  
  This file contains Danish samples that have been filtered to remove content classified as "Problematic". Filtering is performed by the `filter_danish_data.py` script, which uses a list of Danish inappropriate words and spam-like patterns. Only clean, non-problematic educational data is included.

- **danish_filtered_labelled_data.json**  
  Contains Danish samples from the filtered dataset, each labeled with an educational score and justification. The labels were generated using the Google Gemini Flash 2.5 API, as implemented in `get_labels.py`. For each text, the API provides:
    - An educational score (1-5) based on a detailed rubric.
    - A short justification for the assigned score.
    - The full API response for reference.

---

### English Data

- **english_classified_data.csv**  
  Contains English text samples classified using the `fineweb-edu-classifier` model, based on the FineWeb dataset. Each entry includes the model's classification score and relevant metadata. The classification process is implemented in `fineweb_edu_classification.py`.

- **english_fineweb_merged_data.csv**  
  This file is a merged dataset combining:
    - English samples classified with the fineweb-edu-classifier,
    - Additional samples from the `fineweb-edu-2` and `fineweb-edu` HuggingFace datasets.
  
  The merging and processing logic is implemented in `combined_edu_fineweb_data.py`. The merged dataset provides a comprehensive set of English educational samples with classification scores and metadata for training and evaluation.

---

### Data Processing Scripts

- **get_danish_data.py**  
  Downloads and prepares the raw Danish dataset from the FineWeb source.

- **filter_danish_data.py**  
  Filters the Danish dataset to remove problematic or inappropriate content.

- **get_labels.py**  
  Uses the Gemini API to assign educational scores and justifications to the filtered Danish data.

- **fineweb_edu_classification.py**  
  Classifies English FineWeb samples using the fineweb-edu-classifier model.

- **combined_edu_fineweb_data.py**  
  Merges English classified data with additional educational datasets.

---

### Notes

- All files are in CSV or JSON format for easy loading and processing.
- Filtered datasets are recommended for training, while unfiltered/raw datasets can be used for further data cleaning or analysis.
- For details on how the datasets are processed and labeled, see the scripts in the `data_processing/` directory.

## Data Processing

The `data_processing/` folder contains utilities for preparing training data. Key components include:

- **data_process.py** – utility functions for merging English and Danish datasets.
- **danish_training_data/** – scripts for downloading Danish samples from FineWeb (`get_danish_data.py`), filtering out spam or inappropriate content (`filter_danish_data.py`), and scoring each example using Gemini (`get_labels.py`).
- **english_training_data/** – code for classifying English FineWeb samples (`fineweb_edu_classification.py`) and merging them with additional datasets (`combined_edu_fineweb_data.py`).

## Training

Training scripts live in the `training/` directory. `train.py` loads the merged dataset, splits it into train/validation sets and fine‑tunes a transformer model for regression. Configuration files under `training/config/` specify parameters such as model name and number of samples to use. The empty `evaluate.py` is a placeholder for future evaluation utilities, while `utils.py` provides helper functions like `set_seed`.

## Self Annotation

The `self_annotation/` folder contains notebooks and a small Streamlit tool (`annotation.py`) used to manually label data and measure annotator agreement. These files are intended for exploratory analysis and are not required for basic training runs.

## Archive

Deprecated experiments are stored in the `archive/` directory. They are kept for reference only and are not maintained.

## Requirements and Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

After installing, you can launch a training run with `python training/train.py` (edit the YAML files under `training/config/` as needed).
