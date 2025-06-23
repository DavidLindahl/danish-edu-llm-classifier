import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

# --- Configuration ---
# Ensure these paths are correct for your project structure
MODEL_PREDICTIONS_CSV = "results/test_results_with_predictions.csv"
GEMINI_PREDICTIONS_CSV = "results/gemini_predictions.csv"
HUMAN_ANNOTATIONS_JSON = "src/annotation/annotations_mikkel.json"
GOLD_STANDARD_CSV = "src/annotation/test_final.csv" # Ground truth

SCORE_MAPPING = {'None': 0, 'Minimal': 1, 'Basic': 2, 'Good': 3, 'Excellent': 4}

# --- Output Paths ---
OUTPUT_DIR = "archive/evaluation_results"
SUMMARY_TABLE_PATH = os.path.join(OUTPUT_DIR, "final_summary_table_clas.csv")
CONFUSION_MATRICES_PATH = os.path.join(OUTPUT_DIR, "confusion_matrices_clas.png")
METRICS_BARCHART_PATH = os.path.join(OUTPUT_DIR, "metrics_barchart_clas.png")

# --- Helper Functions ---

def transform_wide_to_long(df_wide, true_label_col='int_score'):
    """
    Transforms the wide-format prediction CSV into a long-format DataFrame.
    """
    all_predictions = []
    pred_cols = [c for c in df_wide.columns if c.startswith('predicted_label_')]
    
    print("Reshaping data for the following models from CSV:", [c.replace('predicted_label_', '') for c in pred_cols])

    for col in pred_cols:
        model_name = col.replace('predicted_label_', '')
        raw_col = f"raw_prediction_{model_name}"
        
        temp_df = df_wide[['text', true_label_col]].copy()
        temp_df['model_name'] = model_name
        temp_df['final_prediction'] = df_wide[col]
        temp_df['raw_prediction'] = df_wide.get(raw_col, np.nan)
        temp_df = temp_df.rename(columns={true_label_col: 'true_label'})
        all_predictions.append(temp_df)
        
    if not all_predictions:
        return pd.DataFrame()
        
    return pd.concat(all_predictions, ignore_index=True)


def plot_confusion_matrices(master_df, summary_df):
    """
    Draws a 3x2 grid of confusion matrices in a fixed order, populating
    all plots for which data is available.
    """
    # --- Identify model names ---
    all_model_names = master_df["model_name"].unique()

    zeroshot_name = next((n for n in all_model_names if "zero" in n.lower()), "N/A")

    fewshot_models = summary_df[
        summary_df['model_name'].str.contains('fewshot', case=False, na=False) &
        ~summary_df['model_name'].str.contains('zero', case=False, na=False)
    ].sort_values('accuracy', ascending=False)

    best_ft_name = fewshot_models.iloc[0]['model_name'] if len(fewshot_models) > 0 else "N/A"
    second_ft_name = fewshot_models.iloc[1]['model_name'] if len(fewshot_models) > 1 else "N/A"
    
    # Note: "Full fine-tune" will show "Data Not Found" if no model name in your
    # data contains both "full" and "finetune". This is expected behavior.
    full_ft_name = next((n for n in all_model_names if "full" in n.lower() and "finetuning" in n.lower()), "N/A")
    
    gemini_name = next((n for n in all_model_names if "gemini" in n.lower()), "N/A")
    human_name = next((n for n in all_model_names if "human" in n.lower()), "N/A")

    # --- Fixed display order and titles ---
    models_to_plot = [
        ("base_zeroshot", zeroshot_name),
        ("2nd best fewshot model", second_ft_name),
        ("Best fewshot model", best_ft_name),
        ("Full fine-tune", full_ft_name),
        ("Gemini 2.5 Flash", gemini_name),
        ("Human_Mikkel", human_name)
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 21))
    axes = axes.flatten()

    for idx, (base_title, model_name) in enumerate(models_to_plot):
        ax = axes[idx]
        subset = master_df[master_df['model_name'] == model_name]
        
        accuracy_val = summary_df.loc[summary_df['model_name'] == model_name, 'accuracy'].values
        acc_str = f"(Acc: {accuracy_val[0]:.3f})" if accuracy_val.size > 0 else "(N/A)"
        
        # --- Generate Final Title ---
        if model_name == "N/A" or subset.empty:
            final_title = f"{base_title} {acc_str}"
        elif base_title == "base_zeroshot":
             final_title = f"{model_name} {acc_str}"
        elif "fewshot" in base_title.lower():
             final_title = f"{base_title} ({model_name}) {acc_str}"
        else:
             final_title = f"{base_title} {acc_str}"

        ax.set_title(final_title, fontsize=14)

        if subset.empty:
            ax.text(0.5, 0.5, 'Data Not Found', ha='center', va='center',
                    transform=ax.transAxes, fontsize=16, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        cm = confusion_matrix(subset['true_label'],
                              subset['final_prediction'],
                              labels=list(range(5)))

        sns.heatmap(cm, ax=ax, annot=True, fmt='d', cmap='Blues',
                    cbar=False, annot_kws={"size": 16})
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)

    plt.tight_layout(pad=3.0)
    plt.savefig(CONFUSION_MATRICES_PATH, dpi=300)
    print(f"✅ Confusion matrices saved to: {CONFUSION_MATRICES_PATH}")
    plt.show()


def plot_metrics_barchart(summary_df):
    """Generates and saves a bar chart comparing key metrics across models."""
    df_melted = summary_df.melt(
        id_vars='model_name', 
        value_vars=['accuracy', 'f1_macro', 'f1_weighted', 'mse'],
        var_name='metric', 
        value_name='score'
    )
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_melted, x='score', y='model_name', hue='metric', orient='h', palette='viridis')
    
    plt.title('Model Performance Comparison', fontsize=16)
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('')
    plt.xlim(0, max(1.0, df_melted['score'].max() * 1.05))
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(METRICS_BARCHART_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ Metrics bar chart saved to: {METRICS_BARCHART_PATH}")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load all data sources
    print("--- 📂 Loading Data ---")
    try:
        df_true = pd.read_csv(GOLD_STANDARD_CSV)
        df_models_wide = pd.read_csv(MODEL_PREDICTIONS_CSV)
        df_gemini = pd.read_csv(GEMINI_PREDICTIONS_CSV)
        df_human = pd.read_json(HUMAN_ANNOTATIONS_JSON)
        human_display_name = "Human_Mikkel"
        print("All data files loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find a required data file: {e.filename}")
        print("Please ensure all file paths at the top of the script are correct.")
        exit()

    # 2. Prepare and merge data into a single long-format DataFrame
    print("\n--- 🔄 Processing and Merging Data ---")
    # Process models from the main CSV
    master_df_from_csv = transform_wide_to_long(df_models_wide)

    # Prepare Gemini and Human data
    df_human['final_prediction'] = df_human['our_label'].map(SCORE_MAPPING)
    df_gemini['final_prediction'] = df_gemini['int_score']

    # Combine all annotators into a single master dataframe
    all_dfs = [master_df_from_csv]
    for df_annotator, name in [(df_gemini, "Gemini 2.5 Flash"), (df_human, human_display_name)]:
        print(f"Processing and merging '{name}'...")
        # Merge with ground truth to align labels using the 'text' column
        merged = pd.merge(
            df_true[['text', 'int_score']].rename(columns={'int_score': 'true_label'}),
            df_annotator[['text', 'final_prediction']],
            on='text',
            how='left'
        )
        merged['model_name'] = name
        merged['raw_prediction'] = merged['final_prediction'] # Use final_prediction for MSE
        all_dfs.append(merged)

    master_df = pd.concat(all_dfs, ignore_index=True).dropna(subset=['final_prediction'])
    master_df['final_prediction'] = master_df['final_prediction'].astype(int)
    master_df['true_label'] = master_df['true_label'].astype(int)
    print("All models merged into a master dataframe.")

    # 3. Calculate and save summary metrics
    print("\n--- 🧮 Calculating Summary Metrics ---")
    rows = []
    for name, g in tqdm(master_df.groupby('model_name'), desc="Calculating Metrics"):
        rows.append({
            'model_name': name,
            'mse': mean_squared_error(g['true_label'], g['raw_prediction']),
            'accuracy': accuracy_score(g['true_label'], g['final_prediction']),
            'f1_macro': f1_score(g['true_label'], g['final_prediction'], average='macro', zero_division=0),
            'f1_weighted': f1_score(g['true_label'], g['final_prediction'], average='weighted', zero_division=0),
        })

    summary_df = pd.DataFrame(rows).sort_values('accuracy', ascending=False).round(4)

    print("\n--- 📊 Final Summary of All Model Results ---")
    print(summary_df.to_string())
    summary_df.to_csv(SUMMARY_TABLE_PATH, index=False)
    print(f"\n✅ Summary table saved to: {SUMMARY_TABLE_PATH}")

    # 4. Generate and save all plots
    print("\n--- 🖼️ Generating Visualizations ---")
    plot_metrics_barchart(summary_df)
    plot_confusion_matrices(master_df, summary_df)

    print("\n--- ✅ Evaluation Complete ---")
    print(f"All outputs saved in the '{OUTPUT_DIR}' directory.")