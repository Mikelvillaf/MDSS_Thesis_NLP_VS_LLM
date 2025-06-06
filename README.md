# Predicting Helpfulness of Amazon Customer Reviews

This project explores the prediction of review helpfulness on Amazon using various machine learning models and large language models (LLMs). It benchmarks traditional feature-based ML classifiers (Random Forest, SVM, Gradient Boosting), deep learning approaches (CNN, RCNN for text-only and hybrid features), and LLM-based classification (tested with models like GPT series via OpenAI API and Gemini series via Google API).

The experiment uses a subset of the [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io). The codebase is designed for modularity, configurability, and reproducibility.

## 🔍 Objective

To determine whether a customer review is **helpful** or **unhelpful** using different modeling approaches. We benchmark the performance of:

-   ML models using structured and/or NLP (TF-IDF) features.
-   Deep learning models (CNN, RCNN) trained on review text, with an option for hybrid models incorporating structured features.
-   LLMs (e.g., GPT series, Gemini series) used as zero-shot or few-shot classifiers via their respective APIs.

## ⚙️ Project Structure

-   `amazon-helpfulness/`
    -   `configs/`              # Configuration files
        -   `experiment_config.yaml` # Main configuration for pipeline runs
        -   `hyperparameters.yaml`   # Tuned hyperparameters for models
    -   `data/`                 # Local review/meta files (not tracked by Git)
    -   `results/`              # Output: metrics, predictions, figures
    -   `eda/`                  # Output: eda figures
    -   `scripts/`              # All processing and modeling logic
        -   `data_loader.py`
        -   `evaluation.py`
        -   `feature_engineering.py`
        -   `label_generation.py`
        -   `llm_prediction.py`
        -   `model_training.py`
        -   `preprocessing.py`
        -   `tune_hyperparameters.py` # Script for Optuna-based tuning
        -   `utils.py`
    -   `main.py`               # Orchestrates full experiment runs
    -   `eda.py`                # Creates eda figures
    -   `requirements.txt`      # Python dependencies

## 📦 Dataset

We use the [Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) by McAuley Lab. Experiments can be configured to run on specific categories (e.g., `CDs_and_Vinyl`) and filtered for a defined year range (e.g., 2012-2022, as set in `experiment_config.yaml`). Data is expected in JSONL format.

### Key Data Fields:

**From Raw Data (Input):**
* `rating` (float): Star rating given by the user.
* `title` (str): Title of the review.
* `text` (str): Main content of the review.
* `helpful_vote` (int): Number of "helpful" votes received (Note: original dataset might use `helpful_votes`).
* `verified_purchase` (bool): Whether the purchase was verified.
* `timestamp` (int): Unix timestamp of the review.
* `parent_asin` (str): ASIN of the parent product.
* `user_id` (str): ID of the reviewing user.
* `price` (float): Product price (loaded from separate metadata files, e.g., `meta_CDs_and_Vinyl.jsonl`).

**Derived During Preprocessing/Feature Engineering:**
* `year` (int): Extracted from `timestamp`.
* `clean_text` (str): Lowercased and stripped review text.
* `clean_title` (str): Lowercased and stripped review title.
* `full_text` (str): Concatenation of `clean_title` and `clean_text`.
* `review_word_count` (int): Number of words in `clean_text`.
* `review_char_count` (int): Number of characters in `clean_text`.
* `total_vote` (int): Sum of `helpful_vote` for all reviews of the same `parent_asin` (calculated conditionally if `labeling_mode` is "threshold").
* `sentiment_score` (float): Sentiment polarity of `full_text` (via TextBlob).
* `verified_encoded` (int): 0 or 1 encoding of `verified_purchase`.

## 🛠️ Pipeline Overview

The `main.py` script orchestrates the end-to-end pipeline, driven by `experiment_config.yaml` and `hyperparameters.yaml`:

1.  **Configuration & Setup:**
    * Loads experiment settings and (tuned) model hyperparameters.
    * Sets up timestamped output directories for results.
2.  **Seed Iteration:** The pipeline runs for one or more specified random seeds for result aggregation.
3.  **Category Iteration:** Within each seed, the pipeline processes each specified product category.
    * **Data Loading (`data_loader.py`):**
        * Loads reviews (`load_reviews`) and product metadata (including price via `load_and_clean_metadata`). Supports limiting initial rows loaded.
    * **Preprocessing (`preprocessing.py`):**
        * Cleans text fields, merges price data.
        * Conditionally calculates `total_vote` per product if labeling mode is "threshold".
        * Derives features like `year`, `full_text`, `review_word_count`.
    * **Label Generation (`label_generation.py`, called via `data_loader.identify_label_candidates`):**
        * Assigns binary helpfulness labels (0 or 1) based on the chosen strategy (see "Labeling Strategy" below).
    * **Temporal Splitting & Sampling (`data_loader.py`):**
        * Splits data into training, validation (optional), and test sets based on review years (`create_balanced_temporal_splits`).
        * Supports balanced or imbalanced sampling strategies per split.
    * **Feature Engineering (`feature_engineering.py`):**
        * **ML Features:** Fits a `ColumnTransformer` (`fit_feature_extractor`) on the training set and transforms all splits. Combines `StandardScaler` for structured features and `TfidfVectorizer` for text features based on `feature_set` config (`structured`, `nlp`, `hybrid`). Derived features like sentiment are added.
        * **DL Structured Features:** If DL hybrid mode is enabled (`dl_feature_set: 'hybrid'`), a subset of the engineered ML features are extracted for the deep learning models.
    * **Model Training & Evaluation (`model_training.py`, `llm_prediction.py`, `evaluation.py`):**
        * **ML Models:** Trains and evaluates models specified in `config.models_to_run.ml` (e.g., Random Forest, SVM, Gradient Boosting).
        * **DL Models:** Trains and evaluates Keras-based models specified in `config.models_to_run.dl` (e.g., CNN, RCNN). Supports text-only and hybrid (text + structured features) modes. Uses `EarlyStopping` based on validation performance.
        * **LLM Evaluation:** Evaluates LLMs specified in `config.models_to_run` under provider keys (e.g., `llm_openai`, `llm_google`). Supports zero-shot and few-shot prompting modes. Samples a subset of the test set if configured.
    * Individual model metrics and predictions (optionally with price) are saved per seed.
4.  **Final Summary & Reporting (`evaluation.py`):**
    * Aggregates results from all seeds and categories (`summarize_evaluations`).
    * Generates and saves summary plots:
        * Average confusion matrices for ML/DL and LLM models.
        * F1 score comparison bar chart.
    * Generates and saves summary tables:
        * Overall accuracy by category.
        * Accuracy by price quantile.
        * TPR/TNR by price quantile.
    * Saves the run configuration and used hyperparameters.

## 🧪 Labeling Strategy

A binary label (0: Unhelpful, 1: Helpful) is assigned based on criteria defined in `experiment_config.yaml` (under the `labeling` key). This is handled by `scripts/label_generation.py`.

Supported modes (`labeling.mode`):

-   **`threshold` Mode:** Uses `helpful_vote` and the calculated `total_vote` (sum of `helpful_vote` for all reviews of the same product).
    -   Helpful (1): if `helpful_vote / total_vote` ≥ `labeling.helpful_ratio_min` (e.g., 0.75)
    -   Unhelpful (0): if `helpful_vote / total_vote` ≤ `labeling.unhelpful_ratio_max` (e.g., 0.25)
-   **`percentile` Mode:** Labels based on top/bottom percentiles of `helpful_vote`.
    -   Helpful (1): Reviews in the top `labeling.top_percentile` of `helpful_vote`.
    -   Unhelpful (0): Reviews in the bottom `labeling.bottom_percentile` of `helpful_vote`.

Common filters applied (configurable under `labeling`):

-   `min_total_votes`: Minimum `total_vote` a review's product must have (applied if `total_vote` is calculated). Also, the `generate_labels` function itself can apply a `min_total_votes` filter on individual reviews if the `total_vote` column (pertaining to the specific review, not the product sum) exists from preprocessing, though this specific column isn't explicitly generated in the current `preprocess_reviews`. The primary `min_total_votes` effect is tied to the product sum if using threshold mode or if that column were explicitly added. *Correction based on `generate_labels.py`: `min_total_votes` is applied to the `total_vote` column of the review itself, if that column exists in the input DataFrame to `generate_labels`.*
-   `use_length_filter`: If true, filters reviews based on `min_review_words` and `max_review_words` (applied to `review_word_count`).

Reviews not meeting the criteria for helpful/unhelpful are discarded for training/evaluation.

## ✨ Feature Engineering Details

Handled by `scripts/feature_engineering.py`:

-   **Derived Features (`_add_derived_features`):**
    -   `verified_encoded`: Binary encoding of `verified_purchase`.
    -   `sentiment_score`: Polarity score from TextBlob applied to `full_text`.
    -   `review_word_count` and `review_char_count` are recalculated on `clean_text`.
-   **ML Feature Extractor (`_create_featurizer`, `fit_feature_extractor`):**
    -   Uses `sklearn.compose.ColumnTransformer`.
    -   **Structured Features:** `rating`, `verified_encoded`, `review_word_count`, `review_char_count`, `sentiment_score` are scaled using `StandardScaler`.
    -   **Text Features:** `full_text` is vectorized using `TfidfVectorizer` (configurable `text_max_features`).
    -   The combination of features used is controlled by `feature_set` in `experiment_config.yaml` (`structured`, `nlp`, or `hybrid`).
-   **DL Hybrid Features:**
    -   If `dl_feature_set` in `experiment_config.yaml` is `hybrid`, the first `dl_num_structured_features` columns from the ML features (after transformation by the `ColumnTransformer`) are used as structured input alongside text embeddings for DL models.

The fitted `ColumnTransformer` (featurizer) is saved for each seed and category.

## 🤖 Models Implemented

The project supports training and evaluating several types of models:

**1. Machine Learning (ML) Models (`scripts/model_training.py`):**
    * **Random Forest:** `RandomForestClassifier`
    * **Support Vector Machine (SVM):** `LinearSVC`
    * **Gradient Boosting Machine (GBM):** `GradientBoostingClassifier`
    * Hyperparameters for these models are loaded from `hyperparameters.yaml`.

**2. Deep Learning (DL) Models (`scripts/model_training.py`):**
    * Built using TensorFlow/Keras.
    * **CNN (Convolutional Neural Network):** A 1D CNN over text embeddings.
    * **RCNN (Recurrent Convolutional Neural Network):** An LSTM layer followed by a 1D CNN over text embeddings.
    * **Modes:**
        * **Text-Only:** Uses only text embeddings as input.
        * **Hybrid:** Concatenates text features (from CNN/RCNN layers) with pre-engineered structured features before the final dense layers. Enabled by `dl_feature_set: 'hybrid'`.
    * Hyperparameters (e.g., `embedding_dim`, `conv1d_filters`, `lstm_units`, `dense_units`, `dropout_cat`, `learning_rate_pow`) are a combination of defaults in `main.py` and overrides from `hyperparameters.yaml`.

**3. Large Language Models (LLMs) (`scripts/llm_prediction.py`):**
    * Accessed via API.
    * **Providers & Wrappers:**
        * **OpenAI:** `OpenAIWrapper` (e.g., for "gpt-3.5-turbo", "gpt-4o-mini"). Requires `OPENAI_API_KEY`.
        * **Google:** `GoogleWrapper` (e.g., for "gemini-1.5-pro", "gemini-2.0-flash-lite"). Requires Google API key (env var name configurable, e.g., `GOOGLE_API_KEY`).
    * **Prompting Modes (configurable `prompting_modes`):**
        * `zero_shot`: LLM classifies based on the review text and a generic instruction.
        * `few_shot`: LLM is provided with a few examples of reviews and their labels before classifying the target review. Example selection strategy is configurable (`llm_evaluation.few_shot.example_selection_strategy`).
    * Prompt templates are configurable in `experiment_config.yaml`.

## 📊 Evaluation Metrics

Models are primarily evaluated using the following metrics, calculated by `scripts/evaluation.py`:

-   **Accuracy**
-   **F1 Score** (Weighted average)
-   **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve): Calculated if model provides probability scores.
-   **Confusion Matrix:** Values (TN, FP, FN, TP) are stored.

**Aggregated Reporting:**
-   The `main.py` script orchestrates the collection of these metrics across multiple random seeds.
-   `summarize_evaluations` calculates average metrics (and implicitly standard deviation through individual run results, though not explicitly printed in the final summary to console).
-   **Price Quantile Analysis (`analyze_price_quantiles`):** Accuracy and CM metrics are also calculated for different product price quantiles if price information is available.
-   **Visualizations:**
    * Average Confusion Matrix plots (one for ML/DL, one for LLMs).
    * F1 Score Comparison bar chart across all models.
-   **Tables (CSV output):**
    * `overall_by_category_table.csv`: Average accuracy per model for each category, and overall.
    * `price_quantile_table.csv`: Average accuracy per model for each price quantile.
    * `price_quantile_cm_metrics_table.csv`: Average TPR and TNR per model for each price quantile.

Inference time and API costs (for LLMs) are important practical considerations but are not automatically tracked by the current `evaluation.py` script.

## <img src="https://avatars.githubusercontent.com/u/124583290?s=200&v=4" alt="Weave logo" width="20"/> Experiment Tracking with Weave (Weights & Biases)

This project uses [Weave](https://wandb.ai/weave) decorators (`@weave.op()`) on key functions within the `scripts/` modules (e.g., `data_loader.py`, `model_training.py`, `evaluation.py`). This enables tracing and logging of data processing steps, function inputs/outputs, and model performance with Weights & Biases, facilitating experiment tracking and debugging.

To activate Weave tracking:
1.  Ensure you have a Weights & Biases account.
2.  Log in via the CLI: `wandb login`
3.  While `weave.init("project_name")` is not explicitly called in `main.py`, Weave operations can still be tracked. For more structured W&B runs, consider adding `weave.init()` at the beginning of your main script.

## ⚙️ Hyperparameter Tuning

The project includes `scripts/tune_hyperparameters.py` for automated hyperparameter optimization using Optuna.

-   **Process:**
    1.  Loads data for a specified category (hardcoded as `CDs_and_Vinyl` in the script).
    2.  Performs data preparation and feature engineering similar to `main.py`.
    3.  Defines an Optuna `objective` function that trains and evaluates ML or DL models on a validation set using hyperparameters suggested by Optuna.
    4.  Runs a configurable number of trials for each model type.
-   **Supported Models for Tuning:**
    -   ML: SVM, Random Forest, Gradient Boosting.
    -   DL: CNN, RCNN (if TensorFlow is available).
-   **Output:** The best hyperparameters found for each model are saved to `configs/hyperparameters.yaml`. This file is then used by `main.py` during full experiment runs.
-   **Configuration:** Key settings like the category to tune, models, and number of trials are defined as constants within `tune_hyperparameters.py`.

To run tuning: `python scripts/tune_hyperparameters.py`

## 📄 Configuration Files

The pipeline's behavior is primarily controlled by two YAML configuration files in the `configs/` directory:

1.  **`experiment_config.yaml`**:
    * General experiment parameters: `categories` to run, `data_path_template`, `metadata_path_template`, `year_range`.
    * `labeling` strategy: `mode`, `top_percentile`, `bottom_percentile`, `helpful_ratio_min`, `unhelpful_ratio_max`, `min_total_votes`, `use_length_filter`, `min_review_words`, `max_review_words`.
    * `temporal_split_years`: Years for train, validation, and test sets.
    * `balanced_sampling`: `use_strict_balancing`, `samples_per_class` (for train, val, test), `max_total_samples_imbalanced`.
    * ML features: `feature_set` (`hybrid`, `nlp`, `structured`), `text_max_features`.
    * DL features & base hyperparameters: `dl_feature_set` (`hybrid`), `dl_num_structured_features`, `dl_max_words`, `dl_max_len`, `dl_embedding_dim`, `dl_epochs`, `dl_batch_size`, `dl_conv1d_filters`, `dl_lstm_units`, `dl_dense_units`, `dl_dropout_cat`, `dl_learning_rate_pow`.
    * `models_to_run`: Lists of ML (`ml`) and DL (`dl`) model keys, and LLM provider configurations (e.g., `llm_openai: ["gpt-3.5-turbo"]`, `llm_google: ["gemini-1.5-pro"]`).
    * `llm_evaluation`: API key environment variable names, `test_sample_size`, `prompting_modes` (`zero_shot`, `few_shot`), prompt templates, few-shot example selection strategy, request parameters (`request_timeout`, `max_retries`, `retry_delay`).
    * `random_seeds`: List of seeds for multiple runs.
    * `output_dir`: Base directory for results.
    * `hyperparameters_file`: Path to the hyperparameter file (typically `configs/hyperparameters.yaml`).

2.  **`hyperparameters.yaml`**:
    * Stores the (best) hyperparameters for each model type, typically generated by `scripts/tune_hyperparameters.py`.
    * `main.py` loads this file to get specific model parameters, which can override or supplement the base DL hyperparameters defined in `experiment_config.yaml`.

## 💡 Reproducibility

-   **Configuration Driven:** The entire pipeline is controlled by `experiment_config.yaml` and `hyperparameters.yaml`.
-   **Modular Code:** Logic is separated into modules within the `scripts/` directory.
-   **Seeding:** Random seeds for `numpy`, `random`, and TensorFlow (if used) are set via `experiment_config.yaml` and applied in `main.py` for each run, ensuring consistent data splitting, sampling, and model initializations.
-   **Output Management:** Results (metrics, plots, saved featurizers, predictions) are organized into timestamped run directories under `results/`, with subdirectories for each random seed. The configuration used for each run is also saved.

## 📄 Citation

If you use the Amazon Reviews 2023 dataset, please cite the relevant publication from the dataset providers. Check the dataset source on Hugging Face or the McAuley Lab website for the appropriate citation.

The placeholder citation from your original README was:
> Hou, Yupeng, et al. *Bridging Language and Items for Retrieval and Recommendation.* arXiv preprint arXiv:2403.03952, 2024.
*(Please verify and update with the correct citation for the Amazon Reviews 2023 dataset if different.)*

## 🧠 Credits

-   Developed by [Mikel Villalabeitia](https://github.com/Mikelvillaf)
-   Dataset by McAuley Lab, UCSD

## 🚀 Getting Started

1.  ### Clone the repository
    ```bash
    git clone [https://github.com/Mikelvillaf/MDSS_Thesis_NLP_VS_LLM.git](https://github.com/Mikelvillaf/MDSS_Thesis_NLP_VS_LLM.git)
    cd MDSS_Thesis_NLP_VS_LLM
    ```

2.  ### Create and activate a virtual environment (recommended)
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    ```

3.  ### Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

4.  ### Set up API Keys (Environment Variables)

    For LLM evaluations, set the necessary API keys as environment variables. The specific environment variable names (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`) are referenced in `configs/experiment_config.yaml` under `llm_evaluation.openai_api_key_env_var` and `llm_evaluation.google_api_key_env_var` respectively.

    **macOS/Linux:**
    ```bash
    export OPENAI_API_KEY='your-actual-openai-api-key'
    export GOOGLE_API_KEY='your-actual-google-api-key'
    ```

    **Windows (Command Prompt):**
    ```bash
    set OPENAI_API_KEY=your-actual-openai-api-key
    set GOOGLE_API_KEY=your-actual-google-api-key
    ```
    (Alternatively, set them system-wide or via your IDE's environment variable settings.)

5.  ### Configure Experiment

    Modify `configs/experiment_config.yaml` (or create a copy) to define your experiment settings:
    -   `categories`: List of dataset categories to process (e.g., `["CDs_and_Vinyl"]`).
    -   `data_path_template`, `metadata_path_template`: Paths to your data files. Ensure they point to your local copies of the Amazon Reviews 2023 dataset files (e.g., `data/CDs_and_Vinyl.jsonl`).
    -   `year_range`: Specify the start and end years for filtering reviews.
    -   `max_initial_rows_per_category`: Optionally limit the number of rows initially loaded per category for faster runs or memory management.
    -   `labeling` section:
        -   `mode`: `"threshold"` or `"percentile"`.
        -   Parameters for the chosen mode (e.g., `helpful_ratio_min`, `unhelpful_ratio_max` for threshold; `top_percentile`, `bottom_percentile` for percentile).
        -   `min_total_votes`: Minimum votes on a review for it to be considered for labeling.
        -   `use_length_filter`, `min_review_words`, `max_review_words`: For filtering reviews by word count.
    -   `temporal_split_years`: Define years for `train_years`, `val_year`, and `test_year`.
    -   `balanced_sampling`:
        -   `use_strict_balancing`: True for balanced classes, False for potentially imbalanced.
        -   `samples_per_class`: Target number of samples per class for each split (if `use_strict_balancing` is true).
        -   `max_total_samples_imbalanced`: Maximum total samples per split (if `use_strict_balancing` is false).
    -   ML features:
        -   `feature_set`: `"hybrid"`, `"nlp"`, or `"structured"`.
        -   `text_max_features`: Max features for TF-IDF.
    -   DL features & base hyperparameters:
        -   `dl_feature_set`: `"hybrid"` (to use structured features alongside text) or other (implicitly text-only).
        -   `dl_num_structured_features`: Number of structured features to use if `dl_feature_set` is `hybrid`.
        -   `dl_max_words`, `dl_max_len`, `dl_embedding_dim`, `dl_epochs`, `dl_batch_size`, `dl_conv1d_filters`, `dl_lstm_units`, `dl_dense_units`, `dl_dropout_cat`, `dl_learning_rate_pow`. These are base values; tuned values from `hyperparameters.yaml` will take precedence for specific models.
    -   `models_to_run`: Specify which ML (`ml`), DL (`dl`), and LLM models (e.g., `llm_openai: ["gpt-3.5-turbo"]`, `llm_google: ["gemini-1.5-pro"]`) to include.
    -   `llm_evaluation` section: LLM provider details, specific model IDs, API key environment variable names, `test_sample_size`, `prompting_modes` (`zero_shot`, `few_shot`), prompt templates, few-shot example selection strategy, request parameters (`request_timeout`, `max_retries`, `retry_delay`).
    -   `random_seeds`: List of integer seeds for multiple runs to ensure reproducibility and allow for averaging results.
    -   `output_dir`: Base directory where all results will be saved.
    -   `hyperparameters_file`: Path to the YAML file containing tuned hyperparameters (typically `configs/hyperparameters.yaml`).

6.  ### (Optional) Tune Hyperparameters

    If you want to find optimal hyperparameters for your models on a specific category before running the main pipeline:
    ```bash
    python scripts/tune_hyperparameters.py
    ```
    This script uses Optuna to perform hyperparameter optimization. It focuses on the `CDs_and_Vinyl` category by default (can be changed in the script) and saves the best found parameters to `configs/hyperparameters.yaml`. This file is then used by `main.py`.

7.  ### Run Pipeline & Check Results

    Execute the main pipeline script:
    ```bash
    python main.py
    ```
    Results, including metrics (JSON files), plots (PNG files), saved featurizers (Joblib files), and detailed predictions (CSV files), will be stored in the directory specified by `output_dir` in your configuration, under a timestamped run folder (e.g., `results/run_YYYYMMDD_HHMMSS/`). Within this, there will be subfolders for each random seed. The configuration files used for the run (`config_used.yaml`, `hyperparameters_used.yaml`) are also saved in the main run directory.