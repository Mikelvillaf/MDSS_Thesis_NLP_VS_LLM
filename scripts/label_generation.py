import pandas as pd
import weave

@weave.op()
def generate_labels(
    df: pd.DataFrame,
    mode: str = "percentile",
    top_percentile: float = 0.2,
    bottom_percentile: float = 0.4,
    helpful_ratio_min: float = 0.75,
    unhelpful_ratio_max: float = 0.35,
    min_total_votes: int = 10
) -> pd.DataFrame:
    df = df.copy()

    if mode == "threshold":
        if "helpful_vote" not in df.columns or "total_vote" not in df.columns:
            raise ValueError("Missing helpful_vote or total_vote column for threshold labeling")

        df = df[df["total_vote"] >= min_total_votes]
        df["helpful_ratio"] = df["helpful_vote"] / df["total_vote"]

        # ✅ NOW it's safe to print
        print("📊 Helpful ratio stats:")
        print(df["helpful_ratio"].describe())
        print("✅ Helpful (>= min):", (df["helpful_ratio"] >= helpful_ratio_min).sum())
        print("✅ Unhelpful (<= max):", (df["helpful_ratio"] <= unhelpful_ratio_max).sum())
        print("⚠️ Between thresholds:", ((df["helpful_ratio"] > unhelpful_ratio_max) & (df["helpful_ratio"] < helpful_ratio_min)).sum())
        print("❌ NaNs in helpful_ratio:", df["helpful_ratio"].isna().sum())

        df["label"] = None
        df.loc[df["helpful_ratio"] >= helpful_ratio_min, "label"] = 1
        df.loc[df["helpful_ratio"] <= unhelpful_ratio_max, "label"] = 0

        labeled_df = df[df["label"].isin([0, 1])].copy()
        labeled_df["label"] = labeled_df["label"].astype(int)

    elif mode == "percentile":
        df = df.sort_values("helpful_vote", ascending=False).reset_index(drop=True)
        total = len(df)
        top_n = int(total * top_percentile)
        bottom_n = int(total * bottom_percentile)

        df["label"] = None
        df.iloc[:top_n, df.columns.get_loc("label")] = 1
        df.iloc[-bottom_n:, df.columns.get_loc("label")] = 0

        labeled_df = df[df["label"].isin([0, 1])].copy()
        labeled_df["label"] = labeled_df["label"].astype(int)

    else:
        raise ValueError("Invalid labeling mode. Choose 'percentile' or 'threshold'.")

    print(f"✅ Labeled dataset: {len(labeled_df)} rows "
        f"(Helpful: {(labeled_df['label'] == 1).sum()}, "
        f"Unhelpful: {(labeled_df['label'] == 0).sum()})")

    return labeled_df