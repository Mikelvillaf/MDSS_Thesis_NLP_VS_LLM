# scripts/evaluation.py

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import math # For calculating subplot layout
import weave
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)
import pandas as pd
from typing import Optional, List, Dict, Any
from collections import defaultdict

@weave.op()
def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "model", # Full name like Category_ModelIdentifier_seedX
    output_dir: str = "results/",
    category: Optional[str] = None, # Added
    model_identifier: Optional[str] = None # Added (e.g., cnn_Hybrid, random_forest)
) -> dict:
    """Calculates Acc, F1, ROC AUC, CM. Saves results & CM plot."""
    os.makedirs(output_dir, exist_ok=True)
    # Store category and model_identifier in results if provided
    results = {
        "full_model_name_with_seed": model_name, # Original name for traceability
        "category": category,
        "model_identifier": model_identifier
    }
    cm_val = None # Renamed from cm to avoid conflict with confusion_matrix function

    try:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        results["f1_score"] = report.get("weighted avg", {}).get("f1-score", 0.0)
        results["accuracy"] = accuracy_score(y_true, y_pred)
        results["classification_report"] = report
    except Exception as e:
        print(f"❌ Err calc metrics for {model_name}: {e}")
        results["f1_score"]=0.0; results["accuracy"]=0.0; results["classification_report"]={"error": str(e)}

    results["roc_auc"] = None
    unique_true_labels = np.unique(y_true)
    if y_proba is not None and len(unique_true_labels) > 1:
        try:
            results["roc_auc"] = roc_auc_score(y_true, np.asarray(y_proba))
        except ValueError: # Handles cases with only one class in y_true for roc_auc
             pass # Keep as None
        except Exception as e: print(f"⚠️ ROC AUC err for {model_name}: {e}")


    # Use a unique filename for the CM plot based on the full model_name
    cm_plot_filename = f"{model_name}_confusion_matrix.png"
    cm_path = os.path.join(output_dir, cm_plot_filename)
    results["confusion_matrix_values"] = None
    try:
        cm_val = confusion_matrix(y_true, y_pred, labels=[0, 1]) # Ensure labels=[0, 1]
        results["confusion_matrix_values"] = cm_val.tolist() # Store as list for JSON
        results["_cm_numpy"] = cm_val # Store numpy array for internal aggregation
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=[0, 1])
        disp.plot(cmap="Blues"); plt.title(f'CM - {model_name}'); plt.savefig(cm_path); plt.close()
    except Exception as e: print(f"❌ Error generating/saving CM for {model_name}: {e}"); plt.close()

    # Ensure roc_auc is JSON serializable (None or float)
    results["roc_auc"] = results["roc_auc"] if pd.notna(results["roc_auc"]) else None

    # Use a unique filename for metrics JSON based on the full model_name
    metrics_json_filename = f"{model_name}_metrics.json"
    results_path = os.path.join(output_dir, metrics_json_filename)
    try:
        # Prepare a copy for JSON serialization, removing non-serializable numpy CM
        json_save_results = results.copy()
        if "_cm_numpy" in json_save_results:
            del json_save_results["_cm_numpy"]

        def convert_np_types(o):
            if isinstance(o, np.generic): return o.item()
            if o is None: return "N/A" # Or however you want to represent None in JSON
            raise TypeError
        with open(results_path, "w") as f: json.dump(json_save_results, f, indent=4, default=convert_np_types)
    except Exception as e: print(f"❌ Error saving metrics JSON for {model_name}: {e}")

    # Handle roc_auc display for print
    roc_auc_display = results['roc_auc'] if results['roc_auc'] is not None else "N/A"
    if isinstance(roc_auc_display, float): roc_auc_display = f"{roc_auc_display:.4f}"

    print(f"📊 Metrics {model_name}: Acc={results['accuracy']:.4f}, F1={results.get('f1_score',0.0):.4f}, ROC AUC={roc_auc_display}")
    return results


@weave.op()
def summarize_evaluations(all_results_data: List[Dict], result_dir: str) -> Dict[str, Any]:
    """
    Aggregates metrics from all_results_data.
    Organizes by model_identifier, then by category, and also provides overall averages.
    Saves a summary.json to result_dir.
    """
    print(f"\n--- Summarizing All Results ---")
    if not all_results_data:
        print("⚠️ No results data provided to summarize.")
        return {}

    # Intermediate storage for aggregation
    # model_metrics_aggregated: {
    #   model_identifier: {
    #     "all_accuracies": [], "all_f1_scores": [], "all_roc_aucs": [], "all_cms": [], "total_runs": 0,
    #     "categories": {
    #       category_name: {"accuracies": [], "f1_scores": [], "roc_aucs": [], "cms": [], "num_seeds": 0}
    #     }
    #   }
    # }
    model_metrics_aggregated = defaultdict(lambda: {
        "all_accuracies": [], "all_f1_scores": [], "all_roc_aucs": [], "all_cms": [], "total_runs": 0,
        "categories": defaultdict(lambda: {"accuracies": [], "f1_scores": [], "roc_aucs": [], "cms": [], "num_seeds": 0})
    })

    for res in all_results_data:
        model_id = res.get("model_identifier")
        category = res.get("category")

        if not model_id or not category:
            print(f"⚠️ Skipping result due to missing model_identifier or category: {res.get('full_model_name_with_seed')}")
            continue

        acc = res.get("accuracy")
        f1 = res.get("f1_score")
        roc_auc = res.get("roc_auc") # This might be None or "N/A" from JSON
        cm_numpy = res.get("_cm_numpy") # Try to get the numpy version first
        if cm_numpy is None and res.get("confusion_matrix_values"): # Fallback to list
            try:
                cm_numpy = np.array(res["confusion_matrix_values"])
                if cm_numpy.shape != (2,2): cm_numpy = None
            except: cm_numpy = None


        # Overall aggregation for the model_identifier
        if acc is not None: model_metrics_aggregated[model_id]["all_accuracies"].append(acc)
        if f1 is not None: model_metrics_aggregated[model_id]["all_f1_scores"].append(f1)
        if roc_auc is not None and pd.notna(roc_auc) and not isinstance(roc_auc, str):
             model_metrics_aggregated[model_id]["all_roc_aucs"].append(float(roc_auc))
        if cm_numpy is not None: model_metrics_aggregated[model_id]["all_cms"].append(cm_numpy)
        model_metrics_aggregated[model_id]["total_runs"] += 1

        # Per-category aggregation for the model_identifier
        cat_agg = model_metrics_aggregated[model_id]["categories"][category]
        if acc is not None: cat_agg["accuracies"].append(acc)
        if f1 is not None: cat_agg["f1_scores"].append(f1)
        if roc_auc is not None and pd.notna(roc_auc) and not isinstance(roc_auc, str):
             cat_agg["roc_aucs"].append(float(roc_auc))
        if cm_numpy is not None: cat_agg["cms"].append(cm_numpy)
        cat_agg["num_seeds"] += 1

    final_summary = {}
    for model_id, data in model_metrics_aggregated.items():
        final_summary[model_id] = {
            "metrics_across_all_categories_and_seeds": {
                "avg_accuracy": np.mean(data["all_accuracies"]) if data["all_accuracies"] else None,
                "avg_f1": np.mean(data["all_f1_scores"]) if data["all_f1_scores"] else None,
                "avg_roc_auc": np.mean(data["all_roc_aucs"]) if data["all_roc_aucs"] else None,
                "_avg_cm_ndarray": np.mean(np.array(data["all_cms"]), axis=0) if data["all_cms"] else None,
                "total_runs": data["total_runs"]
            },
            "metrics_per_category": {}
        }
        for cat, cat_data in data["categories"].items():
            final_summary[model_id]["metrics_per_category"][cat] = {
                "avg_accuracy": np.mean(cat_data["accuracies"]) if cat_data["accuracies"] else None,
                "avg_f1": np.mean(cat_data["f1_scores"]) if cat_data["f1_scores"] else None,
                "avg_roc_auc": np.mean(cat_data["roc_aucs"]) if cat_data["roc_aucs"] else None,
                "_avg_cm_ndarray": np.mean(np.array(cat_data["cms"]), axis=0) if cat_data["cms"] else None,
                "num_seeds": cat_data["num_seeds"]
            }
    summary_path = os.path.join(result_dir, "summary_new_structure.json") # Save with new name for inspection
    try:
        # Prepare for JSON: convert ndarrays to lists, handle None
        json_serializable_summary = {}
        for model_id, model_data_val in final_summary.items():
            json_serializable_summary[model_id] = {"metrics_across_all_categories_and_seeds": {}, "metrics_per_category": {}}
            # Overall metrics
            overall_metrics = model_data_val["metrics_across_all_categories_and_seeds"]
            json_serializable_summary[model_id]["metrics_across_all_categories_and_seeds"] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else (float(v) if isinstance(v, (np.floating, np.integer)) else v))
                for k, v in overall_metrics.items()
            }
            # Per category metrics
            for cat, cat_metrics_val in model_data_val["metrics_per_category"].items():
                json_serializable_summary[model_id]["metrics_per_category"][cat] = {
                    k_cat: (v_cat.tolist() if isinstance(v_cat, np.ndarray) else (float(v_cat) if isinstance(v_cat, (np.floating, np.integer)) else v_cat))
                    for k_cat, v_cat in cat_metrics_val.items()
                }

        with open(summary_path, "w") as f:
            json.dump(json_serializable_summary, f, indent=4, default=lambda o: o if o is not None else "N/A")
        print(f"📊 New Detailed Summary saved to: {summary_path}")
    except Exception as e:
        print(f"⚠️ Error saving new_structure_summary.json: {e}")
        import traceback
        traceback.print_exc()

    return final_summary


def analyze_price_quantiles(
    main_run_dir: str,
    num_quantiles: int = 4,
    known_model_identifiers: List[str] = None # Added parameter
) -> Dict[str, Any]:
    """
    Loads detailed predictions, calculates accuracy & CM per price quantile.
    Uses known_model_identifiers for robust parsing of category and model_identifier from filenames.
    """
    print(f"\n--- Price Quantile Analysis (Quantiles: {num_quantiles}) ---")
    if known_model_identifiers is None:
        known_model_identifiers = []
        print("   ⚠️ Warning: known_model_identifiers not provided to analyze_price_quantiles. Parsing may be less robust.")

    prediction_files = glob.glob(os.path.join(main_run_dir, "seed_*", "*_predictions.csv"), recursive=True)

    if not prediction_files:
        print("   ⚠️ No prediction files found. Skipping price quantile analysis.")
        return {}

    all_preds_list = []
    print(f"   Found {len(prediction_files)} prediction files. Loading...")
    for f_path in prediction_files:
        try:
            preds_df = pd.read_csv(f_path)
            if not all(col in preds_df.columns for col in ['y_true', 'y_pred', 'price']):
                 print(f"   ⚠️ Skipping {os.path.basename(f_path)}: Missing required columns (y_true, y_pred, price).")
                 continue

            filename = os.path.basename(f_path) # e.g., CDs_and_Vinyl_random_forest_seed42_predictions.csv
            # Extract the part before "_seedXX_predictions.csv"
            # Example: "CDs_and_Vinyl_random_forest"
            name_before_seed_suffix = filename.split('_seed')[0]

            category_parsed = None
            model_identifier_parsed = None

            # Try to parse using known_model_identifiers (sorted by length descending)
            for m_id in known_model_identifiers:
                # Check if name_before_seed_suffix ends with _{m_id}
                # This assumes the model_identifier is appended with an underscore
                if name_before_seed_suffix.endswith(f"_{m_id}"):
                    # Category is everything before this _{m_id}
                    potential_cat = name_before_seed_suffix[:-len(f"_{m_id}")]
                    if potential_cat: # Ensure category part is not empty
                        category_parsed = potential_cat
                        model_identifier_parsed = m_id
                        break
                # Check if name_before_seed_suffix IS m_id (e.g. category is empty, model is "svm")
                # This case is less likely if categories always exist.
                elif name_before_seed_suffix == m_id:
                    category_parsed = "" # Or some placeholder for empty category
                    model_identifier_parsed = m_id
                    break
            
            if not model_identifier_parsed:
                print(f"   ⚠️ Could not parse category/model from '{name_before_seed_suffix}' using known identifiers. Check filename structure or known_model_identifiers list. Skipping {filename}.")
                continue
            
            # The group key for combined_preds_df should be the original, full name_before_seed_suffix
            # as it represents the unique Category-Model combination for that seed's predictions.
            preds_df['category_model_identifier_group_key'] = name_before_seed_suffix
            preds_df['model_identifier_parsed'] = model_identifier_parsed # This is the "pure" model id
            preds_df['category_parsed'] = category_parsed # This is the "pure" category

            all_preds_list.append(preds_df)
        except Exception as e:
            print(f"   ⚠️ Error loading or processing {os.path.basename(f_path)}: {e}")
            import traceback
            traceback.print_exc()

    if not all_preds_list:
        print("   ❌ No valid prediction data loaded. Cannot perform analysis.")
        return {"error": "No valid prediction data loaded."}

    combined_preds_df = pd.concat(all_preds_list, ignore_index=True)
    # ... (rest of the function: price cleaning, grouping, quantile calculation remains the same,
    #      but it will use 'category_model_identifier_group_key' for grouping the DataFrame,
    #      and 'model_identifier_parsed', 'category_parsed' for the output dict structure)

    analysis_results = {}
    # Group by the 'category_model_identifier_group_key' (e.g., "CDs_and_Vinyl_random_forest")
    # This group (model_df) contains data for one Category-Model combination, aggregated over seeds.
    for group_key, model_df in combined_preds_df.groupby('category_model_identifier_group_key'):
        # All rows in model_df will have the same 'model_identifier_parsed' and 'category_parsed'
        # because they were derived from the filename which is common to the group_key.
        model_id_for_output = model_df['model_identifier_parsed'].iloc[0]
        category_for_output = model_df['category_parsed'].iloc[0]

        print(f"\n   Analyzing model_group: {group_key} ({len(model_df)} predictions with valid price)")
        print(f"     (Category='{category_for_output}', ModelIdentifier='{model_id_for_output}')")

        overall_acc_pf = accuracy_score(model_df['y_true'], model_df['y_pred'])
        overall_cm_pf = confusion_matrix(model_df['y_true'], model_df['y_pred'], labels=[0, 1])

        model_results_entry = {
             'overall_acc_price_filtered': overall_acc_pf,
             'overall_cm_price_filtered': overall_cm_pf.tolist(),
             'quantiles': {},
             'model_identifier': model_id_for_output, # Use parsed pure model_id
             'category': category_for_output      # Use parsed pure category
        }

        if len(model_df) < num_quantiles * 2:
             print(f"      ⚠️ Skipping quantile metrics for {group_key}: Not enough data points ({len(model_df)}) for {num_quantiles} quantiles.")
             analysis_results[group_key] = model_results_entry # Still store overall results
             continue
        # ... (rest of quantile calculation and storing into model_results_entry)
        try:
            quantile_labels = [f"Q{i+1}" for i in range(num_quantiles)]
            # Use qcut on the model-specific price data
            model_df['price_quantile_label'] = pd.qcut(model_df['price'], q=num_quantiles, labels=quantile_labels, duplicates='drop')
            # print(f"      Price Quantile Distribution for {group_key}:\n{model_df['price_quantile_label'].value_counts().sort_index().to_string()}")

            for q_label, quantile_data_group in model_df.groupby('price_quantile_label', observed=False):
                 q_label_str = str(q_label)
                 if not quantile_data_group.empty:
                    q_acc = accuracy_score(quantile_data_group['y_true'], quantile_data_group['y_pred'])
                    q_cm_val = confusion_matrix(quantile_data_group['y_true'], quantile_data_group['y_pred'], labels=[0, 1])
                    model_results_entry['quantiles'][q_label_str] = {
                        'accuracy': q_acc,
                        'cm': q_cm_val.tolist(),
                        'n_samples': len(quantile_data_group)
                    }
                 else:
                    model_results_entry['quantiles'][q_label_str] = {'accuracy': None, 'cm': None, 'n_samples': 0}
            
            analysis_results[group_key] = model_results_entry # Store results under the original group_key

        except ValueError as qcut_error: # Handle error if qcut fails (e.g. not enough unique price points)
            print(f"      ❌ Error calculating quantiles for {group_key}: {qcut_error}. Saving only overall price-filtered results.")
            analysis_results[group_key] = model_results_entry # Store overall results
        except Exception as e: # Catch any other unexpected errors during quantile processing
            print(f"      ❌ Unexpected error during quantile analysis for {group_key}: {e}")
            # Store error information along with parsed identifiers if available
            analysis_results[group_key] = {
                "error": f"Analysis failed: {e}",
                "model_identifier": model_id_for_output,
                "category": category_for_output
            }
            
    if not analysis_results:
         print("   ⚠️ No models could be analyzed in price quantile.")
         return {}

    return analysis_results 


def plot_all_average_cms(
    summary_data: Dict[str, Any],
    main_run_dir: str,
    model_filter_keys: Optional[List[str]] = None,
    model_name_map: Optional[Dict[str, str]] = None,
    plot_filename: str = "overall_summary_avg_cms.png" 
):
    """
    Plots available average confusion matrices from the summary data in subplots.
    Filters by model_filter_keys if provided, uses model_name_map for display names.
    Saves plot to the specified plot_filename.
    """
    print(f"\n--- Plotting Average Confusion Matrices (Output: {plot_filename}) ---")

    models_with_cm_to_plot = []
    
    keys_to_consider = model_filter_keys if model_filter_keys is not None else sorted(summary_data.keys())

    for model_type_key in keys_to_consider:
        if model_type_key not in summary_data:
            if model_filter_keys is not None:
                print(f"   ⚠️ Model type key '{model_type_key}' from filter not found in summary data. Skipping for plot '{plot_filename}'.")
            continue

        model_summary = summary_data[model_type_key]
        overall_metrics = model_summary.get('metrics_across_all_categories_and_seeds', {})
        avg_cm_nd = overall_metrics.get('_avg_cm_ndarray')

        display_name = model_type_key 
        if model_name_map and model_type_key in model_name_map:
            display_name = model_name_map[model_type_key]
        elif model_name_map: 
             print(f"   ⚠️ No display name mapping for '{model_type_key}' for plot '{plot_filename}'. Using original key.")

        if isinstance(avg_cm_nd, np.ndarray) and avg_cm_nd.shape == (2,2):
            models_with_cm_to_plot.append({'name': display_name, 'cm': avg_cm_nd})
        else:
            if model_filter_keys is not None and model_type_key in model_filter_keys:
                 print(f"   ⚠️ No valid overall average CM found for filtered model type '{model_type_key}' (Display: '{display_name}'). Excluded from plot '{plot_filename}'.")

    if not models_with_cm_to_plot:
        print(f"   No valid average CMs found for the current selection to plot in '{plot_filename}'.")
        return

    num_plots = len(models_with_cm_to_plot)
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)
    
    # Adjust figure size to prevent crowding, especially if names are long
    fig_width = max(cols * 4.5, num_plots * 2 if rows == 1 else cols * 4) 
    fig_height = rows * 4 
    if num_plots == 1: # Special case for single plot for better sizing
        fig_width = 5
        fig_height = 4.5


    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()

    for i, model_info_item in enumerate(models_with_cm_to_plot):
        ax = axes[i]
        disp = ConfusionMatrixDisplay(confusion_matrix=model_info_item['cm'], display_labels=[0, 1])
        disp.plot(ax=ax, cmap="Blues", values_format=".1f" if np.any(model_info_item['cm'] >=1) else ".3f")
        # Adjust title font size if needed
        ax.set_title(f"{model_info_item['name']}\n(Avg CM across Cats & Seeds)", fontsize=9 if num_plots > 4 else 10)

    for j in range(i + 1, len(axes)): 
        axes[j].axis('off')

    plt.tight_layout(pad=1.5) # Add some padding
    plot_path = os.path.join(main_run_dir, plot_filename)
    try:
        plt.savefig(plot_path)
        print(f"   ✅ Average CMs plot saved to: {plot_path}")
    except Exception as e:
        print(f"   ❌ Error saving average CMs plot '{plot_path}': {e}")
    plt.close(fig)


def plot_f1_comparison_chart(summary_data: Dict[str, Any], main_run_dir: str):
    """Creates a bar chart comparing model F1 scores, highest on the right."""
    print("\n--- Generating F1 Score Comparison Chart ---")
    model_f1_scores = []
    for model_type, data in summary_data.items():
        overall_metrics = data.get('metrics_across_all_categories_and_seeds', {})
        avg_f1 = overall_metrics.get('avg_f1')
        if avg_f1 is not None:
            model_f1_scores.append({'model_type': model_type, 'avg_f1': avg_f1})
        else:
            print(f"   ⚠️ No F1 score found for model {model_type}, will be excluded from chart.")


    if not model_f1_scores:
        print("   No model F1 scores available to plot.")
        return

    df_f1 = pd.DataFrame(model_f1_scores)
    df_f1 = df_f1.sort_values(by='avg_f1', ascending=True) # Sort for plotting

    plt.figure(figsize=(max(8, len(df_f1) * 0.8), 6)) # Adjust width based on num models
    bars = plt.bar(df_f1['model_type'], df_f1['avg_f1'], color=plt.cm.viridis(np.linspace(0.4, 0.8, len(df_f1))))

    plt.ylabel('Average F1 Score (across Categories & Seeds)')
    plt.xlabel('Model Type')
    plt.title('Model F1 Score Comparison')
    plt.xticks(rotation=45, ha="right")
    plt.ylim(bottom=max(0, df_f1['avg_f1'].min() - 0.05), top=min(1,df_f1['avg_f1'].max() + 0.05)) # Adjust y-lim
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add F1 scores on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)


    plt.tight_layout()
    plot_path = os.path.join(main_run_dir, "model_f1_score_comparison.png")
    try:
        plt.savefig(plot_path)
        print(f"   ✅ Model F1 score comparison chart saved to: {plot_path}")
    except Exception as e:
        print(f"   ❌ Error saving F1 score comparison chart: {e}")
    plt.close()

def plot_f1_comparison_chart(
    summary_data: Dict[str, Any],
    main_run_dir: str,
    model_name_map: Optional[Dict[str, str]] = None,
    plot_filename: str = "model_f1_score_comparison.png"
):
    """
    Creates a bar chart comparing model F1 scores (overall average across categories & seeds).
    Models are sorted by F1 score, with the highest on the far left.
    Uses model_name_map for display names.
    """
    print(f"\n--- Generating F1 Score Comparison Chart (Output: {plot_filename}) ---")
    model_f1_scores = []
    for original_model_key, data in summary_data.items():
        overall_metrics = data.get('metrics_across_all_categories_and_seeds', {})
        avg_f1 = overall_metrics.get('avg_f1')
        
        display_name = original_model_key # Default to original key
        if model_name_map and original_model_key in model_name_map:
            display_name = model_name_map[original_model_key]
        elif model_name_map:
            print(f"   ⚠️ No display name mapping for F1 chart for '{original_model_key}'. Using original key.")

        if avg_f1 is not None:
            model_f1_scores.append({'model_name_display': display_name, 'avg_f1': avg_f1})
        else:
            print(f"   ⚠️ No F1 score found for model '{display_name}' (Original key: {original_model_key}), will be excluded from F1 chart.")

    if not model_f1_scores:
        print(f"   No model F1 scores available to plot in '{plot_filename}'.")
        return

    df_f1 = pd.DataFrame(model_f1_scores)
    # Sort by avg_f1 descending to have the highest F1 score on the left
    df_f1 = df_f1.sort_values(by='avg_f1', ascending=False)

    plt.figure(figsize=(max(8, len(df_f1) * 0.9), 6.5)) # Adjust width based on num models, slightly taller
    bars = plt.bar(df_f1['model_name_display'], df_f1['avg_f1'], color=plt.cm.viridis(np.linspace(0.3, 0.9, len(df_f1))))

    plt.ylabel('Average F1 Score (Overall)', fontsize=12)
    plt.xlabel('Model Type', fontsize=12)
    plt.title('Overall Model F1 Score Comparison', fontsize=14, pad=20)
    # Rotate x-axis labels for better readability if names are long
    plt.xticks(rotation=40, ha="right", fontsize=10) 
    plt.yticks(fontsize=10)

    # Adjust y-axis limits for better visual spacing
    min_f1 = df_f1['avg_f1'].min()
    max_f1 = df_f1['avg_f1'].max()
    plt.ylim(bottom=max(0, min_f1 - 0.05 if pd.notna(min_f1) else 0), 
            top=min(1, max_f1 + 0.05 if pd.notna(max_f1) else 1))
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add F1 scores on top of bars
    for bar in bars:
        yval = bar.get_height()
        if pd.notna(yval): # Ensure yval is not NaN before formatting
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.3f}', 
                    ha='center', va='bottom', fontsize=9, weight='bold')

    plt.tight_layout(pad=1.5) # Adjust layout to prevent labels from overlapping
    plot_path = os.path.join(main_run_dir, plot_filename)
    try:
        plt.savefig(plot_path)
        print(f"   ✅ Model F1 score comparison chart saved to: {plot_path}")
    except Exception as e:
        print(f"   ❌ Error saving F1 score comparison chart '{plot_path}': {e}")
    plt.close()


def generate_accuracy_table(
    summary_data: Optional[Dict],
    analysis_results: Optional[Dict],
    main_run_dir: str,
    analysis_type: str,
    model_name_map: Optional[Dict[str, str]] = None # New parameter
):
    """
    Generates accuracy tables and saves them.
    Uses model_name_map to rename model identifiers for display in tables.
    """
    print(f"\n--- Accuracy Table Generation (Type: {analysis_type}) ---")
    output_file_path = os.path.join(main_run_dir, f"{analysis_type}_table.csv")

    if analysis_type == 'overall_by_category':
        if not summary_data or ("error" in summary_data and len(summary_data) == 1):
            print("   (No valid summary data for 'overall_by_category' table)")
            return
        try:
            table_data = defaultdict(dict)
            all_original_model_types = sorted(list(summary_data.keys()))
            all_categories = set()

            for original_model_type, model_details in summary_data.items():
                display_model_name = model_name_map.get(original_model_type, original_model_type) if model_name_map else original_model_type
                for cat, cat_metrics in model_details.get("metrics_per_category", {}).items():
                    all_categories.add(cat)
                    table_data[cat][display_model_name] = cat_metrics.get("avg_accuracy")
                overall_model_acc = model_details.get("metrics_across_all_categories_and_seeds", {}).get("avg_accuracy")
                table_data["Overall"][display_model_name] = overall_model_acc
            
            all_categories_sorted = sorted(list(all_categories))
            if "Overall" not in all_categories_sorted and "Overall" in table_data:
                 all_categories_sorted.append("Overall")

            df = pd.DataFrame.from_dict(table_data, orient='index')
            # Ensure column order matches the order of original keys, but with new names
            mapped_column_order = [model_name_map.get(mt, mt) if model_name_map else mt for mt in all_original_model_types]
            # Filter for columns that actually exist in df after mapping, and maintain order
            df = df.reindex(columns=[name for name in mapped_column_order if name in df.columns])
            df = df.reindex(all_categories_sorted)

            # Formatting for display (original df is saved with raw numbers)
            df_display = df.copy()
            for col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            
            print(f"   Overall by Category Accuracy Table (Rows: Categories, Columns: Models):")
            print(df_display.to_string())
            df.to_csv(output_file_path)
            print(f"   ✅ Overall by category table saved to: {output_file_path}")

        except Exception as e:
            print(f"   ❌ Error generating 'overall_by_category' table: {e}")
            import traceback
            traceback.print_exc()

    elif analysis_type == 'price_quantile':
        if not analysis_results or ("error" in analysis_results and len(analysis_results) == 1):
            print("   (No valid price quantile analysis results for table)")
            return
        try:
            quantile_agg_data = defaultdict(list)
            all_quantiles = set()
            all_original_model_identifiers_pq = set()

            for cat_model_id_key, results_entry in analysis_results.items():
                if "error" in results_entry: continue
                original_model_id = results_entry.get('model_identifier')
                if not original_model_id: continue
                all_original_model_identifiers_pq.add(original_model_id)
                quantiles_data = results_entry.get('quantiles', {})
                for q_label, q_metrics in quantiles_data.items():
                    all_quantiles.add(q_label)
                    if q_metrics.get('accuracy') is not None:
                        quantile_agg_data[(original_model_id, q_label)].append(q_metrics['accuracy'])
            
            if not quantile_agg_data:
                 print("   (No valid quantile accuracy data to aggregate).")
                 return

            avg_quantile_accuracies = {key: np.mean(val) for key, val in quantile_agg_data.items()}
            idx = pd.MultiIndex.from_tuples(avg_quantile_accuracies.keys(), names=['original_model_identifier', 'quantile'])
            s = pd.Series(avg_quantile_accuracies.values(), index=idx)
            quantile_df = s.unstack(level='original_model_identifier')

            if model_name_map:
                quantile_df.columns = [model_name_map.get(col, col) for col in quantile_df.columns]
            
            quantile_df = quantile_df.reindex(sorted(list(all_quantiles)))
            # Sort columns by mapped names if map provided, else by original
            sorted_display_cols = sorted(quantile_df.columns)
            quantile_df = quantile_df.reindex(columns=sorted_display_cols)

            quantile_df_display = quantile_df.copy()
            for col in quantile_df_display.columns:
                quantile_df_display[col] = quantile_df_display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

            print("   Price Quantile Accuracy Table (Rows: Quantiles, Columns: Models, Averaged over Categories & Seeds):")
            print(quantile_df_display.to_string())
            quantile_df.to_csv(output_file_path)
            print(f"   ✅ Price quantile accuracy table saved to: {output_file_path}")

        except Exception as e:
            print(f"   ❌ Error generating price quantile accuracy table: {e}")
            import traceback
            traceback.print_exc()
            
    elif analysis_type == 'price_quantile_cm_metrics':
        if not analysis_results or ("error" in analysis_results and len(analysis_results) == 1):
            print("   (No valid price quantile analysis results for CM metrics table)")
            return
        try:
            temp_cm_metrics_agg = defaultdict(list)
            all_metric_cols_cm = set()
            all_original_model_ids_cm = set()

            for cat_model_id_key, results in analysis_results.items():
                if "error" in results: continue
                original_model_id = results.get('model_identifier')
                if not original_model_id: continue
                all_original_model_ids_cm.add(original_model_id)
                # ... (logic to populate temp_cm_metrics_agg as before, using original_model_id) ...
                # Ensure this part uses original_model_id for keys in temp_cm_metrics_agg
                overall_cm_list = results.get('overall_cm_price_filtered')
                if overall_cm_list and isinstance(overall_cm_list, list):
                    try:
                        tn, fp, fn, tp = np.array(overall_cm_list).ravel()
                        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                        temp_cm_metrics_agg[(original_model_id, 'Overall', 'TPR')].append(tpr)
                        temp_cm_metrics_agg[(original_model_id, 'Overall', 'TNR')].append(tnr)
                    except Exception: pass 

                quantile_results_data = results.get('quantiles', {})
                for q_label, q_data_item in quantile_results_data.items():
                    q_cm_list = q_data_item.get('cm')
                    if q_cm_list and isinstance(q_cm_list, list):
                        try:
                            tn, fp, fn, tp = np.array(q_cm_list).ravel()
                            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                            temp_cm_metrics_agg[(original_model_id, q_label, 'TPR')].append(tpr)
                            temp_cm_metrics_agg[(original_model_id, q_label, 'TNR')].append(tnr)
                        except Exception: pass
                
            final_cm_metrics_data_for_df = defaultdict(dict)
            for (original_model_id, period, metric_type), values in temp_cm_metrics_agg.items():
                col_name = f"{period}_{metric_type}"
                all_metric_cols_cm.add(col_name)
                # Use original_model_id as key for now, will rename index later
                final_cm_metrics_data_for_df[original_model_id][col_name] = np.mean(values) if values else None
            
            if not final_cm_metrics_data_for_df:
                print("   (No CM metric data could be extracted/aggregated)")
                return

            cm_metrics_df = pd.DataFrame.from_dict(final_cm_metrics_data_for_df, orient='index')
            
            if model_name_map: # Rename the index (model names)
                cm_metrics_df.index = [model_name_map.get(idx, idx) for idx in cm_metrics_df.index]
            
            cm_metrics_df = cm_metrics_df.sort_index() # Sort rows (mapped model names)
            sorted_cols_cm = sorted(list(all_metric_cols_cm), key=lambda c: (not c.startswith('Overall'), c))
            cm_metrics_df = cm_metrics_df.reindex(columns=sorted_cols_cm)

            cm_metrics_df_display = cm_metrics_df.copy()
            for col in cm_metrics_df_display.columns:
                cm_metrics_df_display[col] = cm_metrics_df_display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

            print("   Price Quantile CM Metrics (TPR/TNR) Table (Averaged over Categories & Seeds):")
            print(cm_metrics_df_display.to_string())
            cm_metrics_df.to_csv(output_file_path)
            print(f"   ✅ Price quantile CM metrics table saved to: {output_file_path}")

        except Exception as e:
            print(f"   ❌ Error generating price quantile CM metrics table: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"   ⚠️ Unknown analysis_type for table generation: {analysis_type}")