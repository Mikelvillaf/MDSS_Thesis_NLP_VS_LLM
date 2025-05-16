# eda.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml # For loading config

# Attempt to import from your project structure
try:
    from scripts.utils import load_config
    from scripts.data_loader import load_reviews, load_and_clean_metadata
    from scripts.preprocessing import preprocess_reviews
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure eda.py is in the project root directory and PYTHONPATH is set up if needed.")
    print("Or, ensure 'scripts' directory is accessible.")
    exit()

# Define output directory for EDA results
EDA_OUTPUT_DIR = "eda"
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)

# --- Plotting Helper Functions ---

def plot_voting_distribution_generic(df, group_col_name, plot_title_prefix, output_filename_prefix):
    if 'helpful_vote' not in df.columns:
        print(f"⚠️ 'helpful_vote' column not found in DataFrame for {plot_title_prefix}. Skipping plot.")
        return

    # Your updated bins for 5 categories
    bins = [-float('inf'), 0, 5, 10, 50, float('inf')]
    labels = [
        "Reviews With Zero Votes",
        "Reviews from 1-5 Votes",
        "Reviews from 5-10 Votes",
        "Reviews from 10-50 Votes",
        "Reviews With 50+ Votes"
    ]
    df['helpful_vote_numeric'] = pd.to_numeric(df['helpful_vote'], errors='coerce').fillna(0)
    df['vote_bin'] = pd.cut(df['helpful_vote_numeric'], bins=bins, labels=labels, right=True)

    # Sort unique groups for consistent plot order
    unique_groups = sorted(df[group_col_name].astype(str).unique()) if group_col_name in df else []


    num_groups = len(unique_groups)

    if num_groups == 0:
        print(f"⚠️ No groups found for {group_col_name} in {plot_title_prefix}. Skipping plot.")
        return

    ncols = 2 if num_groups > 1 else 1
    nrows = (num_groups + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.5 * ncols, 6.5 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, group_value in enumerate(unique_groups):
        ax = axes[i]
        group_df = df[df[group_col_name] == group_value]

        if group_df.empty:
            ax.text(0.5, 0.5, 'No data for this group', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f"{plot_title_prefix}: {group_value}")
            ax.set_xticks([]) 
            ax.set_yticks([]) 
            continue

        counts = group_df['vote_bin'].value_counts().reindex(labels, fill_value=0)
        percentages = (counts / counts.sum() * 100).fillna(0)

        # FIX for FutureWarning from sns.barplot
        if not counts.empty:
            bars = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, ax=ax, palette="Blues_d", legend=False, dodge=False)
        else:
            # Fallback for safety, though group_df.empty should catch this
            ax.text(0.5, 0.5, 'No binned data', ha='center', va='center', transform=ax.transAxes)


        ax.set_title(f"{plot_title_prefix}: {group_value} ({len(group_df):,} Reviews)")
        ax.set_ylabel("Number of Reviews")
        ax.set_xlabel("Helpful Vote Bins")

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        if not counts.empty and 'bars' in locals(): # Check if bars was created
            for bar_idx, bar_item in enumerate(bars.patches):
                y_val = bar_item.get_height()
                percentage = percentages.iloc[bar_idx]
                text_offset = counts.max() * 0.02 if counts.max() > 0 else y_val * 0.05 # Adjust offset if max is 0
                ax.text(bar_item.get_x() + bar_item.get_width()/2.,
                        y_val + text_offset,
                        f'{int(y_val):,}\n({percentage:.0f}%)',
                        ha='center', va='bottom', fontsize=8) # Reduced font size a bit

            ax.set_ylim(0, counts.max() * 1.20) # Increased space for text annotations

        ax.grid(axis='y', linestyle='--', alpha=0.7)

    for j in range(i + 1, len(axes)): # i should be defined from the loop
        fig.delaxes(axes[j])

    plt.suptitle(f"Voting Distributions: {plot_title_prefix}", fontsize=16, y=1.03, weight='bold') # Adjusted y for suptitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect for suptitle
    output_path = os.path.join(EDA_OUTPUT_DIR, f"{output_filename_prefix}_voting_distribution.png")
    plt.savefig(output_path)
    print(f"✅ Saved plot: {output_path}")
    plt.close(fig)


# --- Main EDA Logic ---
def perform_eda(config_path="configs/experiment_config.yaml"):
    print("--- Starting EDA Script ---")
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"❌ Configuration file not found at {config_path}. Exiting.")
        return
    except Exception as e:
        print(f"❌ Error loading configuration: {e}. Exiting.")
        return

    categories_to_run = config.get("categories", [])
    if not categories_to_run:
        print("⚠️ No categories specified in config. Exiting EDA.")
        return

    data_path_template = config.get("data_path_template", "data/{category}.jsonl")
    metadata_path_template = config.get("metadata_path_template", "data/meta_{category}.jsonl")
    year_range = config.get("year_range")
    max_initial_rows = config.get("max_initial_rows_per_category")

    all_categories_df_list = []

    for category in categories_to_run:
        print(f"\n🔄 Processing category: {category}")
        review_file_path = data_path_template.format(category=category)
        metadata_file_path = metadata_path_template.format(category=category)

        print(f"   Loading reviews for {category}...")
        df_raw = load_reviews(filepath=review_file_path, year_range=year_range, max_initial_load=max_initial_rows)
        if df_raw.empty:
            print(f"   ⚠️ No review data loaded for {category}. Skipping.")
            continue

        print(f"   Loading metadata for {category}...")
        meta_df = load_and_clean_metadata(metadata_file_path)

        labeling_config_main = config.get("labeling", {})
        labeling_mode_main = labeling_config_main.get("mode", "percentile")

        print(f"   Preprocessing reviews for {category} (labeling_mode for preprocessing: {labeling_mode_main})...")
        df_processed = preprocess_reviews(df_raw.copy(), metadata_df=meta_df, labeling_mode=labeling_mode_main)
        del df_raw

        if df_processed.empty:
            print(f"   ⚠️ Data empty after preprocessing for {category}. Skipping.")
            continue
        if 'helpful_vote' not in df_processed.columns:
            print(f"   ⚠️ 'helpful_vote' missing for {category} post-preprocessing. Skipping plots for this category.")
            continue

        df_processed['category'] = category
        all_categories_df_list.append(df_processed)

    if not all_categories_df_list:
        print("\n❌ No data processed. EDA finished.")
        return

    combined_df = pd.concat(all_categories_df_list, ignore_index=True)
    combined_df['helpful_vote'] = pd.to_numeric(combined_df['helpful_vote'], errors='coerce').fillna(0)
    if 'year' not in combined_df.columns and 'timestamp' in combined_df.columns:
        print("   Manually creating 'year' column for EDA from 'timestamp'.")
        combined_df["year"] = pd.to_datetime(combined_df["timestamp"], unit="ms", errors='coerce').dt.year
    elif 'year' not in combined_df.columns:
        print("   ⚠️ 'year' column (and 'timestamp') not found. Year-based plots will be skipped.")

    print(f"\n📊 Combined DataFrame for EDA: {len(combined_df)} reviews.")

    # --- Generate Requested Plots ---
    print("\nGenerating voting distributions by category...")
    plot_voting_distribution_generic(
        combined_df.copy(), group_col_name='category',
        plot_title_prefix='Category', output_filename_prefix='by_category'
    )

    if 'price' in combined_df.columns:
        print("\nGenerating voting distribution by price quantile...")
        combined_df['price_numeric'] = pd.to_numeric(combined_df['price'], errors='coerce')
        df_for_quantile = combined_df.dropna(subset=['price_numeric']).copy()
        if not df_for_quantile.empty and df_for_quantile['price_numeric'].nunique() >= 4:
            try:
                # FIX for Price Quantile Labels and FutureWarning
                quantile_custom_labels = [
                    "Q1 (Lowest Prices)",
                    "Q2 (Low-Mid Prices)",
                    "Q3 (Mid-High Prices)",
                    "Q4 (Highest Prices)"
                ]
                df_for_quantile.loc[:, 'price_quantile_label'] = pd.qcut(
                    df_for_quantile['price_numeric'],
                    q=4,
                    labels=quantile_custom_labels, # Use custom string labels
                    duplicates='drop'
                )
                # The .astype(str) line is no longer needed here as labels are already strings

                plot_voting_distribution_generic(
                    df_for_quantile, # Pass the modified df_for_quantile
                    group_col_name='price_quantile_label',
                    plot_title_prefix='Price Quantile',
                    output_filename_prefix='by_price_quantile'
                )
            except ValueError as e: # Catch specific qcut error
                print(f"   ⚠️ Could not create price quantiles (Error: {e}). Skipping. Check price distribution or try fewer quantiles.")
            except Exception as e: # Catch any other unexpected error
                print(f"   ⚠️ An unexpected error occurred during price quantile plot (Error: {e}). Skipping.")
        elif df_for_quantile.empty:
            print("   ⚠️ No data with valid prices for quantiles. Skipping price quantile plot.")
        else:
            print(f"   ⚠️ Not enough unique price points ({df_for_quantile['price_numeric'].nunique()}) for 4 quantiles. Skipping price quantile plot.")
    else:
        print("\n⚠️ 'price' column not found in combined data. Skipping price quantile plot.")


    # --- Additional Plots from Original EDA Notebook ---

    # # Overall Helpful vote distribution (already included as an example previously)
    # print("\nGenerating overall helpful vote distribution...")
    # plt.figure(figsize=(10, 6))
    # sns.histplot(combined_df['helpful_vote'], bins=50, log_scale=(False, True)) # Using already numeric 'helpful_vote'
    # plt.title("Overall Distribution of Helpful Votes (All Processed Categories)")
    # plt.xlabel("Helpful Votes")
    # plt.ylabel("Frequency (log scale)")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(EDA_OUTPUT_DIR, "overall_helpful_vote_distribution.png"))
    # print(f"✅ Saved plot: {os.path.join(EDA_OUTPUT_DIR, 'overall_helpful_vote_distribution.png')}")
    # plt.close()

    # # Proportion of reviews with 0 helpful votes (overall)
    # if not combined_df.empty:
    #     zero_helpful = (combined_df['helpful_vote'] == 0).sum()
    #     total_reviews = len(combined_df)
    #     print(f"📊 Overall: Reviews with 0 helpful votes: {zero_helpful} / {total_reviews} ({zero_helpful/total_reviews:.2%})")

    # # Reviews per product (parent_asin) - overall
    # if 'parent_asin' in combined_df.columns and not combined_df.empty:
    #     print("\nGenerating reviews per product distribution...")
    #     asin_counts = combined_df['parent_asin'].value_counts()
    #     plt.figure(figsize=(10, 6))
    #     sns.histplot(asin_counts, bins=50, log_scale=(True, True))
    #     plt.title("Distribution of Reviews per Product (parent_asin) - Overall")
    #     plt.xlabel("Number of Reviews per Product (log scale)")
    #     plt.ylabel("Number of Products (log scale)")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(EDA_OUTPUT_DIR, "overall_reviews_per_product.png"))
    #     print(f"✅ Saved plot: {os.path.join(EDA_OUTPUT_DIR, 'overall_reviews_per_product.png')}")
    #     plt.close()

    # # Helpfulness vote concentration by product - overall
    # if 'parent_asin' in combined_df.columns and not combined_df.empty:
    #     print("\nGenerating helpful votes per review by product distribution...")
    #     asin_helpful_sum = combined_df.groupby('parent_asin')['helpful_vote'].sum()
    #     asin_review_counts = combined_df['parent_asin'].value_counts()
    #     concentration_df = pd.DataFrame({
    #         'review_count': asin_review_counts,
    #         'total_helpful_votes': asin_helpful_sum
    #     }).fillna(0)
    #     # Avoid division by zero if a product somehow has 0 reviews but appears in groupby (should not happen with value_counts)
    #     concentration_df = concentration_df[concentration_df['review_count'] > 0]
    #     concentration_df['helpful_per_review'] = concentration_df['total_helpful_votes'] / concentration_df['review_count']
        
    #     plt.figure(figsize=(10, 6))
    #     sns.histplot(concentration_df['helpful_per_review'], bins=50, log_scale=(False, True))
    #     plt.title("Distribution of Avg Helpful Votes per Review (by Product) - Overall")
    #     plt.xlabel("Average Helpful Votes per Review for a Product")
    #     plt.ylabel("Number of Products (log scale)")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(EDA_OUTPUT_DIR, "overall_helpful_per_review_by_product.png"))
    #     print(f"✅ Saved plot: {os.path.join(EDA_OUTPUT_DIR, 'overall_helpful_per_review_by_product.png')}")
    #     plt.close()

    # # Pareto Insight: Skewness of Helpful Votes - overall
    # if not combined_df.empty and combined_df['helpful_vote'].sum() > 0:
    #     print("\nGenerating Pareto insight for helpful votes...")
    #     sorted_votes = combined_df["helpful_vote"].sort_values(ascending=False).reset_index(drop=True)
    #     cumulative_votes = sorted_votes.cumsum()
    #     total_votes_sum = sorted_votes.sum() # Ensure this is not zero
        
    #     cumulative_share = cumulative_votes / total_votes_sum if total_votes_sum > 0 else np.zeros_like(cumulative_votes)
    #     percent_reviews = np.arange(1, len(sorted_votes) + 1) / len(sorted_votes)

    #     plt.figure(figsize=(10, 6))
    #     sns.lineplot(x=percent_reviews, y=cumulative_share, color="darkblue", linewidth=2)
    #     plt.axhline(0.8, color="red", linestyle="--", label="80% Helpful Votes")
    #     plt.axvline(0.2, color="green", linestyle="--", label="20% Reviews")
    #     plt.xlabel("Fraction of Reviews (sorted by helpfulness)")
    #     plt.ylabel("Cumulative Share of Total Helpful Votes")
    #     plt.title("Cumulative Distribution of Helpful Votes (Pareto Insight) - Overall")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(EDA_OUTPUT_DIR, "overall_pareto_helpful_votes.png"))
    #     print(f"✅ Saved plot: {os.path.join(EDA_OUTPUT_DIR, 'overall_pareto_helpful_votes.png')}")
    #     plt.close()
    # elif combined_df.empty or combined_df['helpful_vote'].sum() == 0:
    #     print("   ⚠️ Skipping Pareto plot due to empty data or zero total helpful votes.")


    # # Helpful votes per product by year - Boxplot (overall)
    # if 'year' in combined_df.columns and 'parent_asin' in combined_df.columns and not combined_df.empty:
    #     print("\nGenerating helpful votes per product by year boxplot...")
    #     grouped_by_year_asin = combined_df.groupby(["year", "parent_asin"])["helpful_vote"].sum().reset_index()
    #     plt.figure(figsize=(14, 7))
    #     sns.boxplot(data=grouped_by_year_asin, x="year", y="helpful_vote", showfliers=False, palette="coolwarm")
    #     plt.yscale("log")
    #     plt.title("Distribution of Total Helpful Votes per Product, by Year - Overall")
    #     plt.xlabel("Review Year")
    #     plt.ylabel("Total Helpful Votes per Product (log scale)")
    #     plt.xticks(rotation=45, ha='right')
    #     plt.grid(axis='y', linestyle='--', alpha=0.7)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(EDA_OUTPUT_DIR, "overall_helpful_votes_per_product_by_year.png"))
    #     print(f"✅ Saved plot: {os.path.join(EDA_OUTPUT_DIR, 'overall_helpful_votes_per_product_by_year.png')}")
    #     plt.close()

    # # Total helpful votes by year - Barplot (overall)
    # if 'year' in combined_df.columns and not combined_df.empty:
    #     print("\nGenerating total helpful votes by year barplot...")
    #     yearly_total_helpful_votes = combined_df.groupby("year")["helpful_vote"].sum().reset_index()
    #     plt.figure(figsize=(12, 6))
    #     sns.barplot(data=yearly_total_helpful_votes, x="year", y="helpful_vote", palette="viridis")
    #     plt.yscale("log") # Keep log scale if values vary widely
    #     plt.title("Total Helpful Votes by Review Year - Overall")
    #     plt.ylabel("Total Helpful Votes (log scale)")
    #     plt.xlabel("Review Year")
    #     plt.xticks(rotation=45, ha='right')
    #     plt.grid(axis="y", linestyle="--", alpha=0.7)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(EDA_OUTPUT_DIR, "overall_total_helpful_votes_by_year.png"))
    #     print(f"✅ Saved plot: {os.path.join(EDA_OUTPUT_DIR, 'overall_total_helpful_votes_by_year.png')}")
    #     plt.close()

    print("\n--- EDA Script Finished ---")

if __name__ == "__main__":
    perform_eda()