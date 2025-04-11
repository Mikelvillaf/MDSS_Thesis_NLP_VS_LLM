import pandas as pd
from tqdm import tqdm
import os
import weave
from tracking.wandb_init import init_tracking

# --- CONFIG ---
INPUT_FILE = "data/Books.jsonl"
OUTPUT_FILE = "data/Books_labeled.parquet"
HELPFUL_THRESHOLD = 0.75
UNHELPFUL_THRESHOLD = 0.35

@weave.op()
def load_data(filepath):
    return pd.read_json(filepath, lines=True)

@weave.op()
def generate_labels(df):
    df = df.dropna(subset=['helpful_votes', 'parent_asin'])
    df = df[df['helpful_votes'] > 0]

    total_votes_per_product = df.groupby('parent_asin')['helpful_votes'].sum().to_dict()

    tqdm.pandas(desc="Calculating helpfulness ratios")
    df['helpful_ratio'] = df.progress_apply(
        lambda row: row['helpful_votes'] / total_votes_per_product.get(row['parent_asin'], 1),
        axis=1
    )

    def assign_label(ratio):
        if ratio >= HELPFUL_THRESHOLD:
            return 1
        elif ratio <= UNHELPFUL_THRESHOLD:
            return 0
        else:
            return None

    df['label'] = df['helpful_ratio'].apply(assign_label)
    df = df.dropna(subset=['label'])

    return df

def main():
    init_tracking("amazon-helpfulness")  # 🐝 Start tracking

    print(f"📂 Loading: {INPUT_FILE}")
    df = load_data(INPUT_FILE)

    print("⚙️  Generating helpful/unhelpful labels...")
    labeled_df = generate_labels(df)

    print(f"💾 Saving labeled data to: {OUTPUT_FILE}")
    labeled_df.to_parquet(OUTPUT_FILE, index=False)

    print(f"✅ Done! {len(labeled_df)} labeled reviews written.")

if __name__ == "__main__":
    main()