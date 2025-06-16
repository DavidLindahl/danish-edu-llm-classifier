import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix

import krippendorff


# --- Configuration ---
MODEL_PREDICTIONS_CSV = "test/test_results_with_predictions.csv"
GEMINI_PREDICTIONS_CSV = "eval/gemini_predictions.csv"
HUMAN_ANNOTATIONS_JSON = "self_annotation/annotations_david.json"
GOLD_STANDARD_CSV = "self_annotation/test_final.csv" # The new ground truth
SCORE_MAPPING = {'None': 0, 'Minimal': 1, 'Basic': 2, 'Good': 3, 'Excellent': 4}

# --- Output Paths ---
OUTPUT_DIR = "test/report_visuals"
SUMMARY_TABLE_PATH = os.path.join(OUTPUT_DIR, "final_summary_table.csv")
PERFORMANCE_PLOT_PATH = os.path.join(OUTPUT_DIR, "performance_curve.png")
CONFUSION_MATRICES_PATH = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
DISTRIBUTION_PLOT_PATH = os.path.join(OUTPUT_DIR, "prediction_distribution.png")


def transform_wide_to_long(df_wide):
    all_predictions = []
    pred_cols = [c for c in df_wide.columns if c.startswith('predicted_label_')]
    
    for col in pred_cols:
        model_name = col.replace('predicted_label_', '')
        raw_col = f"raw_prediction_{model_name}"
        
        temp_df = df_wide[['text', 'true_label']].copy()
        temp_df['model_name'] = model_name
        temp_df['final_prediction'] = df_wide[col]
        temp_df['raw_prediction'] = df_wide.get(raw_col, np.nan)
        all_predictions.append(temp_df)
        
    return pd.concat(all_predictions, ignore_index=True)


def plot_metrics_barchart(summary_df: pd.DataFrame):
    """
    x-axis = models · y-axis = metric scores · hue = metric name
    Shows MSE, accuracy, F1-macro, F1-weighted, Krippendorff’s α.
    """
    metrics_to_show = ['mse', 'accuracy', 'f1_macro', 'f1_weighted', 'alpha']

    melt = summary_df.melt(
        id_vars='model_name',
        value_vars=metrics_to_show,
        var_name='metric',
        value_name='score'
    )

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=melt,
        x='model_name', y='score',
        hue='metric', ax=ax,
        palette='viridis', edgecolor='black'
    )

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Metric value', fontsize=12)
    ax.set_title('Performance of each model across 5 metrics', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
    ax.legend(title='Metric', bbox_to_anchor=(1.03, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(PERFORMANCE_PLOT_PATH, dpi=300)        # we keep same file name
    print(f"Metrics bar-chart saved → {PERFORMANCE_PLOT_PATH}")
    plt.show()


def plot_confusion_matrices(master_df, summary_df):
    best_fs_model_name = summary_df.loc[summary_df[summary_df['model_name'].str.contains('fewshot')]['f1_macro'].idxmax()]['model_name']
    models_to_plot = {
        "Zero-Shot": "xlm-roberta-danish-educational-scorer-zeroshot",
        f"Best Few-Shot ({best_fs_model_name.split('-')[-1]} samples)": best_fs_model_name,
        "Gemini 2.5 Flash": "Gemini 2.5 Flash",
        "Human (David)": "Human (David)"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.flatten()
    
    for i, (title, model_name) in enumerate(models_to_plot.items()):
        ax = axes[i]
        subset = master_df[master_df['model_name'] == model_name]
        cm = confusion_matrix(subset['true_label'], subset['final_prediction'], labels=list(range(5)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False, annot_kws={"size": 14})
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
    fig.tight_layout(pad=3.0) # Add padding between plots
    plt.savefig(CONFUSION_MATRICES_PATH, dpi=300)
    print(f"Confusion matrix plot saved to {CONFUSION_MATRICES_PATH}")
    plt.show()


def plot_prediction_distribution(master_df, df_true):
    """
    Histogram of score frequencies for every source plus the ground truth.
    Ground truth now taken directly from df_true['int_score'].
    """
    #  ground-truth dataframe
    true_df = df_true[['int_score']].rename(columns={'int_score': 'score'})
    true_df['model_name'] = 'True Distribution'

    #  predictions dataframe
    preds_df = master_df[['final_prediction', 'model_name']]\
                 .rename(columns={'final_prediction': 'score'})

    plot_df = pd.concat([preds_df, true_df], ignore_index=True)
    plot_df['score'] = plot_df['score'].astype(int)

    plt.figure(figsize=(16, 8))
    sns.countplot(data=plot_df, x='score', hue='model_name', palette='viridis')
    plt.title('Distribution of Scores: predictions vs. ground truth', fontsize=16)
    plt.xlabel('Score');  plt.ylabel('Count')
    plt.legend(title='Source', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(DISTRIBUTION_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved → {DISTRIBUTION_PLOT_PATH}")
    plt.show()


def krippendorff_alpha(true_series, pred_series) -> float:
    """
    Two-rater Krippendorff's alpha for ordinal labels 0-4.
    true_series and pred_series must be 1-D arrays of equal length.
    """
    reliability_data = np.vstack([true_series, pred_series])
    return krippendorff.alpha(reliability_data=reliability_data,
                              level_of_measurement='ordinal')

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load ground truth and all predictions
    df_true = pd.read_csv(GOLD_STANDARD_CSV)
    df_models_wide = pd.read_csv(MODEL_PREDICTIONS_CSV)
    df_gemini = pd.read_csv(GEMINI_PREDICTIONS_CSV)
    df_human = pd.read_json(HUMAN_ANNOTATIONS_JSON)

    # 2. Prepare and merge data into a single long-format DataFrame
    df_models_wide = df_models_wide.rename(columns={'int_score': 'true_label'})
    master_df = transform_wide_to_long(df_models_wide)

    df_human['final_prediction'] = df_human['our_label'].map(SCORE_MAPPING)
    df_gemini['final_prediction'] = df_gemini['int_score']

    # Combine all annotators
    all_dfs = [master_df]
    for df_annotator, name in [(df_gemini, "Gemini 2.5 Flash"), (df_human, "Human (David)")]:
        temp_df = df_true[['text', 'int_score']].rename(columns={'int_score': 'true_label'})
        temp_df = temp_df.merge(df_annotator[['text', 'final_prediction']], on='text', how='left')
        temp_df['model_name'] = name
        temp_df['raw_prediction'] = np.nan
        all_dfs.append(temp_df)

    master_df = pd.concat(all_dfs, ignore_index=True).dropna(subset=['final_prediction'])
    master_df['final_prediction'] = master_df['final_prediction'].astype(int)

    # 3. Calculate and save summary metrics
    summary_data = []
    rows = []
    for name, g in master_df.groupby('model_name'):
        # decide which column to use for MSE
        if g['raw_prediction'].notna().any():
            mse_source = 'raw_prediction'
        else:
            mse_source = 'final_prediction'

        rows.append({
            'model_name': name,
            'mse'        : mean_squared_error(g['true_label'], g[mse_source]),
            'accuracy'   : accuracy_score(g['true_label'], g['final_prediction']),
            'f1_macro'   : f1_score(g['true_label'], g['final_prediction'],
                                    average='macro', zero_division=0),
            'f1_weighted': f1_score(g['true_label'], g['final_prediction'],
                                    average='weighted', zero_division=0),
            'alpha'      : krippendorff_alpha(g['true_label'].values,
                                              g['final_prediction'].values)
        })

    summary_df = pd.DataFrame(rows).sort_values('f1_macro', ascending=False).round(4)

    print("--- 📊 Final Summary of All Model Results ---")
    print(summary_df.to_string())
    summary_df.to_csv(SUMMARY_TABLE_PATH, index=False)
    print(f"\n✅ Summary table saved to: {SUMMARY_TABLE_PATH}")

    # 4. Generate and save all plots
    plot_metrics_barchart(summary_df)
    plot_confusion_matrices(master_df, summary_df)
    plot_prediction_distribution(master_df, df_true)

    print("\n--- Evaluation Complete ---")