# Evaluation
# Evaluation

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import weave
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

@weave.op()
def evaluate_model(y_true, y_pred, y_proba=None, model_name="model", output_dir="results/"):
    os.makedirs(output_dir, exist_ok=True)

    report = classification_report(y_true, y_pred, output_dict=True)
    f1 = report["weighted avg"]["f1-score"]

    roc = None
    if y_proba is not None:
        try:
            roc = roc_auc_score(y_true, y_proba)
        except:
            pass

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    results = {
        "model": model_name,
        "f1_score": f1,
        "roc_auc": roc,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    results_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Evaluation saved: {results_path}")
    return results

@weave.op()
def summarize_evaluations(result_dir="results/"):
    from collections import defaultdict

    scores = defaultdict(list)

    for file in os.listdir(result_dir):
        if file.endswith("_metrics.json"):
            with open(os.path.join(result_dir, file), "r") as f:
                res = json.load(f)
                model = res["model"].split("_seed")[0]  # Normalize model name
                scores[model].append(res)

    summary = {}
    for model, runs in scores.items():
        avg_f1 = np.mean([r["f1_score"] for r in runs])
        avg_roc = np.mean([r["roc_auc"] for r in runs if r["roc_auc"] is not None])
        summary[model] = {
            "avg_f1": avg_f1,
            "avg_roc_auc": avg_roc
        }

    summary_path = os.path.join(result_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"📊 Summary saved to: {summary_path}")
    return summary