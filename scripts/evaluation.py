# scripts/evaluation.py (Final Clean Version - Corrected Typo)

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
    accuracy_score
)
from collections import defaultdict
import pandas as pd

@weave.op()
def evaluate_model(y_true, y_pred, y_proba=None, model_name="model", output_dir="results/"):
    """
    Evaluates model performance, saves metrics and confusion matrix.
    """
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    # --- Calculate Metrics ---
    try:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        f1 = report.get("weighted avg", {}).get("f1-score", 0.0)
        acc = accuracy_score(y_true, y_pred)
    except Exception as e:
        print(f"❌ Error calculating classification report/accuracy for {model_name}: {e}")
        report = {"error": str(e)}
        f1 = 0.0
        acc = 0.0

    roc = None
    unique_true_labels = np.unique(y_true)
    if y_proba is not None and len(unique_true_labels) > 1:
        try:
            y_proba = np.asarray(y_proba)
            roc = roc_auc_score(y_true, y_proba)
        except ValueError as e:
            print(f"⚠️ Could not calculate ROC AUC for {model_name}: {e}")
            roc = None
        except Exception as e:
             print(f"❌ Unexpected error calculating ROC AUC for {model_name}: {e}")
             roc = None
    elif len(unique_true_labels) <= 1:
        # This print is expected if only one class in test set for a specific run
        # print(f"ℹ️ ROC AUC requires samples from >1 class for {model_name}. Got {len(unique_true_labels)}.")
        roc = None

    # --- Confusion Matrix ---
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    cm = None # Initialize cm
    try:
        cm = confusion_matrix(y_true, y_pred)
        unique_all_labels = np.unique(np.concatenate((y_true, y_pred)))
        display_labels = sorted(list(unique_all_labels))
        if len(display_labels) == 0: display_labels = [0, 1]
        elif len(display_labels) == 1:
             other_label = 1 if display_labels[0] == 0 else 0
             display_labels = sorted(display_labels + [other_label])

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap="Blues")
        plt.savefig(cm_path)
        plt.close()
        print(f"   Confusion matrix saved to {cm_path}")
    except Exception as e:
        print(f"❌ Error generating/saving confusion matrix for {model_name}: {e}")
        plt.close()


    # --- Save Results ---
    results = {
        "model": model_name,
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": roc if roc is not None else "N/A", # Save as "N/A" for JSON if None
        "classification_report": report,
        "confusion_matrix_values": cm.tolist() if cm is not None else "Error generating CM"
    }

    results_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    try:
        with open(results_path, "w") as f:
            # Helper to convert numpy types during JSON dump
            def convert_np_types(o):
                if isinstance(o, np.generic): return o.item()
                raise TypeError
            json.dump(results, f, indent=4, default=convert_np_types)
        print(f"✅ Evaluation saved: {results_path}")
    except Exception as e:
        print(f"❌ Error saving metrics JSON for {model_name}: {e}")


    print(f"📊 Metrics for {model_name}: Accuracy={acc:.4f}, F1={f1:.4f}, ROC AUC={roc if roc is not None else 'N/A'}")
    return results


@weave.op()
def summarize_evaluations(result_dir="results/"):
    """
    (Clean Version)
    Summarizes evaluation metrics by reading *_metrics.json files recursively
    from the result_dir and averaging scores grouped by base model name using Pandas.
    """
    all_results = []
    print(f"   Scanning for '*_metrics.json' files recursively in: {result_dir}")

    for root, _, files in os.walk(result_dir):
        for file in files:
            if file.endswith("_metrics.json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as f:
                        res = json.load(f)
                        # Basic validation: Check if it's a dict and has 'model' key
                        if isinstance(res, dict) and 'model' in res:
                            all_results.append(res)
                        else:
                            print(f"⚠️ Skipping record from {file_path} (not dict or missing 'model' key).")
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping malformed JSON file: {file_path}")
                except Exception as e:
                    import traceback
                    print(f"⚠️ Unexpected Error processing file {file_path}: {e}")
                    traceback.print_exc()


    print(f"   Finished scanning. Found {len(all_results)} potentially valid records.")
    if not all_results:
        print("⚠️ No result files with basic structure found to summarize.")
        return {}

    # --- Create DataFrame and Aggregate ---
    try:
        results_df = pd.DataFrame(all_results)

        # Define columns expected to be numeric for aggregation
        numeric_cols = ['accuracy', 'f1_score', 'roc_auc']

        # Convert columns to numeric, coercing errors. Replace "N/A" before conversion.
        for col in numeric_cols:
             if col in results_df.columns:
                 # Replace string "N/A" if used during saving
                 results_df[col] = results_df[col].replace("N/A", np.nan)
                 results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
             else:
                 print(f"   Warning: Metric column '{col}' not found in all results. Summary for it will be NaN.")
                 results_df[col] = np.nan # Add missing column as NaN

        # Extract base model name (e.g., "CDs_and_Vinyl_cnn")
        results_df['base_model'] = results_df['model'].str.replace(r'_seed\d+$', '', regex=True)

        # Perform aggregation
        summary = results_df.groupby('base_model').agg(
            avg_accuracy=('accuracy', 'mean'), # mean() automatically skips NaNs
            avg_f1=('f1_score', 'mean'),
            avg_roc_auc=('roc_auc', 'mean'),
            run_count=('model', 'size') # Count how many runs (seeds) were averaged
        ).reset_index()

        # Convert final summary to dictionary
        final_summary = {}
        for item in summary.to_dict(orient='records'):
             model_name = item['base_model']
             final_summary[model_name] = {
                 # Round averages, handle potential NaNs from aggregation
                 'avg_accuracy': round(item['avg_accuracy'], 4) if pd.notna(item['avg_accuracy']) else None,
                 'avg_f1': round(item['avg_f1'], 4) if pd.notna(item['avg_f1']) else None,
                 'avg_roc_auc': round(item['avg_roc_auc'], 4) if pd.notna(item['avg_roc_auc']) else None,
                 'run_count': item['run_count']
             }

        # Save the final summary JSON
        summary_path = os.path.join(result_dir, "summary.json")
        try:
            with open(summary_path, "w") as f:
                 # --- CORRECTED FUNCTION NAME ---
                 def convert_summary_nones(o): # Renamed: No space
                     if o is None: return "N/A" # Or return None for JSON null
                     raise TypeError
                 # --- USE CORRECTED FUNCTION NAME ---
                 json.dump(final_summary, f, indent=4, default=convert_summary_nones)
            print(f"📊 Summary saved to: {summary_path}")
        except Exception as e:
            print(f"⚠️ Error saving summary file: {e}")

        return final_summary

    except Exception as e:
        print(f"❌ Error during DataFrame creation or aggregation: {e}")
        import traceback
        traceback.print_exc()
        return {"error": "Failed during aggregation"}