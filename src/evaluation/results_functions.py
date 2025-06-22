from scipy.stats import bootstrap
from scipy.stats import wilcoxon
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import krippendorff

def Squared_diff_list(score_data): # <-- list of lists

    SD_list = []

    for scores in score_data:
        c = len(scores)
        #The Squared Difference between unordered distinct pairs for one text:
        SD = (2 / (c * (c - 1))) * sum(
            (scores[j] - scores[k]) ** 2 
            for j in range(c) 
            for k in range(j+1,c))
        SD_list.append(SD)

    return SD_list

def hist_dist(sample1, sample2, name1, name2, type):
    sns.set(style="whitegrid")
    
    bins = [0.5 * x - 0.5 for x in range(16)]

    if type == "msd":
        list1 = Squared_diff_list(sample1)
        list2 = Squared_diff_list(sample2)

    elif type == "mean":
        list1 = [np.mean(scores) for scores in sample1]
        if sample2[0] is float:
            list2 = sample2
        else:
            list2 = [np.mean(scores) for scores in sample2]
        
    plt.figure(figsize=(8, 5))
    plt.hist(list1, bins=bins, edgecolor='black', color='skyblue', alpha=0.5, label=name1, rwidth=0.9)
    plt.hist(list2, bins=bins, edgecolor='black', color='salmon', alpha=0.5, label=name2, rwidth=0.9)

    plt.xlabel(type.upper() + ' Values')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

############################ BOOTSTRAP FUNCTIONS ############################
# Plotting bootstrapped MSE for models

def bootstrap_mse_models(model_data, y_human, n_bootstrap=10000, alpha=0.05):

    sns.set(style="whitegrid")

    model_names = []
    mean_errors = []
    cis_lower = []
    cis_upper = []

    for name, predictions in model_data.items():
        preds = np.array(predictions)
        errors = (y_human - preds) ** 2

        model_names.append(name[15:].capitalize())

        res = bootstrap((errors,), np.mean, confidence_level=1 - alpha, 
                        n_resamples=n_bootstrap, method='bca')

        mean_errors.append(np.mean(errors))
        cis_lower.append(res.confidence_interval.low)
        cis_upper.append(res.confidence_interval.high)

    mean_errors = np.array(mean_errors)
    cis_lower = np.array(cis_lower)
    cis_upper = np.array(cis_upper)

    # Compute error bars
    error_lower = mean_errors - cis_lower
    error_upper = cis_upper - mean_errors
    yerr = [error_lower, error_upper]

    # Plotting
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("deep", len(model_names))

    plt.errorbar(model_names, mean_errors, yerr=yerr,
                fmt='o', markersize=8, capsize=6, capthick=2,
                ecolor='gray', color='black', elinewidth=1.5)

    #plt.axhline(min(mean_errors), color='red', linestyle='--', linewidth=1, label='Best model MSE', alpha=0.5)

    plt.ylabel("Mean Squared Error", fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    return plt.show()


# -------------------------------------------------------------------
# Plotting mean differences between models and human annotations

def mean_diff_models_vs_human(data_model_human):
    y_human = np.mean([np.mean(scores) for scores in data_model_human["Human"]])
    mean_diffs = {}
    
    for model_name, model_scores in data_model_human.items():
        if model_name == "Human":
            continue
        y_pred = np.array(model_scores)
        mean_diff = np.mean(y_pred) - y_human
        mean_diffs[model_name] = mean_diff
    
    return mean_diffs


def bootstrap_mean_diff_CI(data_model_human, n_bootstrap=10000, ci=95):
    y_human = np.mean([np.mean(scores) for scores in data_model_human["Human"]])
    ci_dict = {}
    for model_name, model_scores in data_model_human.items():
        if model_name == "Human":
            continue
        model_scores = np.array(model_scores)
        boot_diffs = []
        n = len(model_scores)
        for _ in range(n_bootstrap):
            idx = np.random.randint(0, n, n)
            boot_model = model_scores[idx]
            boot_diff = np.mean(boot_model) - y_human
            boot_diffs.append(boot_diff)
        lower = np.percentile(boot_diffs, (100 - ci) / 2)
        upper = np.percentile(boot_diffs, 100 - (100 - ci) / 2)
        ci_dict[model_name] = (lower, upper)
    return ci_dict



def plot_mean_diff_models_vs_human(data_model_human):

    mean_diffs = mean_diff_models_vs_human(data_model_human)
    mean_diff_cis = bootstrap_mean_diff_CI(data_model_human)

    labels = [name[15:].capitalize() for name in mean_diffs.keys()]
    means = [mean_diffs[name] for name in mean_diffs]
    ci_lowers = [mean_diff_cis[name][0] for name in mean_diffs]
    ci_uppers = [mean_diff_cis[name][1] for name in mean_diffs]

    plt.figure(figsize=(10, 6))  # Wider plot for space
    plt.errorbar(labels, means,
                yerr=[np.array(means) - np.array(ci_lowers),
                    np.array(ci_uppers) - np.array(means)],
                fmt='o', capsize=5, color='black', ecolor='gray', elinewidth=2, markeredgewidth=2)

    plt.axhline(0, color='blue', linestyle='--')
    plt.ylabel('Mean Difference (Model vs. Human)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=11)  # Rotate x-labels for readability
    plt.yticks(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# Plotting bootstrapped MSD for models vs human

def plot_bootstrapped_msd_humanVSmodels(data_model_human, n_bootstrap=10000, alpha=0.05): 
    sns.set(style="whitegrid") 

    model_names = []
    mean_errors = []
    cis_lower = []
    cis_upper = []

    for name, predictions in data_model_human.items():
        model_names.append(name[15:].capitalize() if name != "Human" else "Human")

        if name != "Human":
            combined = [np.append(data_model_human["Human"][i], predictions[i]) for i in range(len(predictions))]
        else:
            combined = data_model_human["Human"]

        y_all = np.asarray(Squared_diff_list(combined))

        res = bootstrap((y_all,), np.mean, confidence_level=1 - alpha, 
                        n_resamples=n_bootstrap, method='BCa')

        mean_errors.append(np.mean(y_all))
        cis_lower.append(res.confidence_interval.low)
        cis_upper.append(res.confidence_interval.high)

    mean_errors = np.array(mean_errors)
    cis_lower = np.array(cis_lower)
    cis_upper = np.array(cis_upper)

    error_lower = mean_errors - cis_lower
    error_upper = cis_upper - mean_errors
    yerr = [error_lower, error_upper]

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    # Plot all except the last point in black
    for i in range(len(model_names) - 1):
        plt.errorbar(model_names[i], mean_errors[i],
                     yerr=[[error_lower[i]], [error_upper[i]]],
                     fmt='o', markersize=8, capsize=6, capthick=2,
                     ecolor='gray', color='black', elinewidth=1.5)

    # Plot the last CI in blue
    i = len(model_names) - 1
    plt.errorbar(model_names[i], mean_errors[i],
                 yerr=[[error_lower[i]], [error_upper[i]]],
                 fmt='o', markersize=8, capsize=6, capthick=2,
                 ecolor='gray', color='blue', elinewidth=1.5)

    plt.ylabel("Mean Squared Difference (MSD)", fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt.show()

# -------------------------------------------------------------------
# Krippendorff's alpha bootstrap function


def bootstrap_alpha(score_data):
    annotations = np.array(score_data)  # shape: (N, 6)

    n_bootstrap = 1000
    alpha_values = []

    for _ in tqdm(range(n_bootstrap)):
        # Sample rows with replacement
        indices = np.random.choice(len(annotations), size=len(annotations), replace=True)
        resampled = annotations[indices]

        # Transpose to shape (n_raters, n_items)
        data = resampled.T

        # Compute Krippendorff's alpha (interval scale assumed here)
        alpha = krippendorff.alpha(reliability_data=data, level_of_measurement='interval')
        alpha_values.append(alpha)

    alpha_values = np.array(alpha_values)

    # Compute 95% CI (percentile method)
    lower = np.percentile(alpha_values, 2.5)
    upper = np.percentile(alpha_values, 97.5)
    mean_alpha = np.mean(alpha_values)

    return [mean_alpha, lower, upper]


def plot_CI(data, labels=None, title="Bootstrapped 95% Confidence Intervals", ylabel="Value"):
    """
    Plot mean and 95% CI for each entry in data.
    Highlights the last CI in blue.
    """

    data = np.array(data)
    means = data[:, 0]
    lowers = data[:, 1]
    uppers = data[:, 2]
    yerr = np.vstack([means - lowers, uppers - means])

    if labels is None:
        labels = [f"Item {i+1}" for i in range(len(means))]

    plt.figure(figsize=(8, 5))

    # Plot all except the last point in black
    for i in range(len(means) - 1):
        plt.errorbar(labels[i], means[i],
                     yerr=[[means[i] - lowers[i]], [uppers[i] - means[i]]],
                     fmt='o', markersize=8, capsize=6, capthick=2,
                     ecolor='gray', color='black', elinewidth=1.5)

    # Plot the last point in blue
    i = len(means) - 1
    plt.errorbar(labels[i], means[i],
                 yerr=[[means[i] - lowers[i]], [uppers[i] - means[i]]],
                 fmt='o', markersize=8, capsize=6, capthick=2,
                 ecolor='gray', color='blue', elinewidth=1.5)

    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()