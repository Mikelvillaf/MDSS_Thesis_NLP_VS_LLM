# scripts/evaluation.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import weave
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score  # <-- Import accuracy_score
)

@weave.op()
def evaluate_model(y_true, y_pred, y_proba=None, model_name="model", output_dir="results/"):
    os.makedirs(output_dir, exist_ok=True)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0) # Added zero_division=0
    f1 = report["weighted avg"]["f1-score"]
    acc = accuracy_score(y_true, y_pred) # <-- Calculate accuracy

    roc = None
    # Ensure there are samples from both classes before calculating ROC AUC
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            roc = roc_auc_score(y_true, y_proba)
        except ValueError as e:
             print(f"⚠️ Could not calculate ROC AUC for {model_name}: {e}")
             roc = None # Set roc to None if calculation fails
    elif len(np.unique(y_true)) <= 1:
         print(f"⚠️ ROC AUC requires samples from more than 1 class. Got {len(np.unique(y_true))} for {model_name}.")
         roc = None # Set roc to None if only one class present

    cm = confusion_matrix(y_true, y_pred)
    # Ensure labels are present for display, handle case with only one class predicted/true
    display_labels = np.unique(np.concatenate((y_true, y_pred)))
    if len(display_labels) == 0: # Handle edge case if y_true/y_pred are empty
        display_labels = [0, 1]
    elif len(display_labels) == 1: # Handle single class case
        display_labels = [display_labels[0], 1 - display_labels[0]] # Add the missing class label


    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap="Blues")
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close() # Close plot to free memory

    results = {
        "model": model_name,
        "accuracy": acc,  # <-- Add accuracy here
        "f1_score": f1,
        "roc_auc": roc,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    results_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Evaluation saved: {results_path}")
    print(f"📊 Metrics for {model_name}: Accuracy={acc:.4f}, F1={f1:.4f}, ROC AUC={roc if roc is None else f'{roc:.4f}'}") # Added print statement
    return results

@weave.op()
def summarize_evaluations(result_dir="results/"):
    from collections import defaultdict
    import pandas as pd # Import pandas for easier handling

    scores = defaultdict(list)
    all_results = []

    for file in os.listdir(result_dir):
        if file.endswith("_metrics.json"):
            try:
                with open(os.path.join(result_dir, file), "r") as f:
                    res = json.load(f)
                    all_results.append(res)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping malformed JSON file: {file}")
            except Exception as e:
                 print(f"⚠️ Error processing file {file}: {e}")


    if not all_results:
        print("⚠️ No valid result files found to summarize.")
        return {}

    # Create DataFrame for easier aggregation
    results_df = pd.DataFrame(all_results)
    # Extract base model name before seed
    results_df['base_model'] = results_df['model'].str.split('_seed').str[0]

    # Group by base model and calculate mean, std dev for numeric metrics
    summary = results_df.groupby('base_model').agg(
        avg_accuracy=('accuracy', 'mean'),
        std_accuracy=('accuracy', 'std'),
        avg_f1=('f1_score', 'mean'),
        std_f1=('f1_score', 'std'),
        avg_roc_auc=('roc_auc', lambda x: x.mean(skipna=True)), # Handle potential NaNs in ROC AUC
        std_roc_auc=('roc_auc', lambda x: x.std(skipna=True))
    ).reset_index()

    # Convert summary to dictionary format if needed, handling NaN std devs
    summary_dict = summary.fillna(0).to_dict(orient='records') # Fill NaN std dev with 0
    final_summary = {item['base_model']: {k: v for k, v in item.items() if k != 'base_model'} for item in summary_dict}


    summary_path = os.path.join(result_dir, "summary.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(final_summary, f, indent=4)
        print(f"📊 Summary saved to: {summary_path}")
    except Exception as e:
        print(f"⚠️ Error saving summary file: {e}")

    return final_summary