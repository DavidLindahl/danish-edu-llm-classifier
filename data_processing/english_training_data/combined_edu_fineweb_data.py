import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import os

HF_DATASET_SCORE_3_NAME = "HuggingFaceFW/fineweb-edu"
HF_CONFIG_SCORE_3 = "CC-MAIN-2024-22"
HF_DATASET_SCORE_2_NAME = "HuggingFaceFW/fineweb-edu-score-2"
HF_CONFIG_SCORE_2 = "CC-MAIN-2024-18"
CSV_FILE_PATH = "../../data/english_fineweb_merged_data.csv"
COMMON_COLUMNS = ["text", "language_score", "token_count", "int_score"]

