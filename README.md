# Predicting Helpfulness of Amazon Customer Reviews

This project explores the prediction of review helpfulness on Amazon using various machine learning models and large language models (LLMs). It compares traditional feature-based ML classifiers (like Random Forest, SVM, and Gradient Boosting), deep learning approaches (CNN/RCNN), and LLM-based classification using GPT or DeepSeekAI.

The experiment uses a subset of the [Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023), which contains over 570 million customer reviews from 1996 to 2023.

## 🔍 Objective

To determine whether a customer review is **helpful** or **unhelpful** using different modeling approaches. We benchmark the performance of:

- Structured + NLP features with classical ML models
- Deep learning models trained directly on review text
- LLMs used as zero-shot classifiers via API

## ⚙️ Project Structure
amazon-helpfulness/
├── data/                 # Local review/meta files (not tracked by Git)
├── scripts/              # All processing and modeling logic
├── tests/                # Unit tests for reproducibility
├── configs/              # Configs for experiments and pipelines
├── results/              # Output: metrics, predictions, figures
├── main.py               # Orchestrates full experiment runs
└── requirements.txt      # Python dependencies

## 📦 Dataset

We use the [Amazon Reviews 2023 dataset](https://github.com/hyp1231/AmazonReviews2023) by McAuley Lab.

Selected product categories include:

- `Books`
- `CDs and Vinyl`
- `Clothing, Shoes, and Jewelry`
- `Beauty and Personal Care`
- `Cell Phones and Accessories`
- `Home and Kitchen`

Each review record includes:

- `rating` (float)
- `title` and `text` (str)
- `helpful_votes` (int)
- `verified_purchase` (bool)
- `timestamp` (int)
- `asin`, `parent_asin`, `user_id` (str)

Metadata includes:

- `price`, `average_rating`, `features`, `description`
- `images`, `store`, `brand`, and category hierarchies

## 🧪 Labeling Strategy

A binary label is assigned to each review:

- **Helpful**: if `helpful_votes / total_votes_on_product` ≥ `0.75`
- **Unhelpful**: if ≤ `0.35`
- All others are discarded to reduce label noise

Reviews with zero votes are excluded.

## 📊 Evaluation Metrics

Models are compared using:

- **F1 Score**
- **ROC-AUC**
- **Inference Time / Cost** (esp. for LLMs)

Experiments are run across multiple seeds and subsets for robustness.

## 💡 Reproducibility

- Modular pipeline with `main.py`
- Seed control via `configs/experiment_config.yaml`
- Unit tests in `tests/`
- Logging and output saved to `results/`

## 📄 Citation

If you use the Amazon Reviews dataset, please cite:

> Hou, Yupeng, et al. *Bridging Language and Items for Retrieval and Recommendation.* arXiv preprint arXiv:2403.03952, 2024.

## 🧠 Credits

Developed by [Mikel Villalabeitia](https://github.com/Mikelvillaf)  
Dataset by McAuley Lab, UCSD