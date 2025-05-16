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

# --- Define Color Palette ---
COLOR_PALETTE = {
    "figure_background": "#FAFBFD",
    "plot_background": "#FFFFFF",
    "data_input_alt": "#EEEEEE",
    "prep_alt": "#DDEEFF",
    "feature_eng_alt": "#CCDDFF",
    "results_viz_alt": "#FFFFCC",
    "primary_bar_color": "#5698C6",
    "secondary_bar_color": "#A8DADC",
    "text_color_dark": "#333333",
    "title_color": "#2c3e50",
    "grid_color": "#D5D8DC",
    "line_plot_main": "#1F77B4",
    "pareto_80_line": "#FF7F0E",
    "pareto_20_line": "#2CA02C",
    "boxplot_palette": "viridis", # Using a seaborn palette for boxplots
    "barplot_palette_year": "viridis"
}

EDA_OUTPUT_DIR = "eda"
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)

# --- Plotting Helper Functions ---

def plot_voting_distribution_generic(df, group_col_name, plot_title_prefix, output_filename_prefix):
    if 'helpful_vote' not in df.columns:
        print(f"⚠️ 'helpful_vote' column not found for {plot_title_prefix}. Skipping.")
        return

    bins = [-float('inf'), 0, 5, 10, 50, float('inf')]
    labels = ["Zero Votes", "1-5 Votes", "5-10 Votes", "10-50 Votes", "50+ Votes"]
    df['helpful_vote_numeric'] = pd.to_numeric(df['helpful_vote'], errors='coerce').fillna(0)
    df['vote_bin'] = pd.cut(df['helpful_vote_numeric'], bins=bins, labels=labels, right=True)

    unique_groups = sorted(df[group_col_name].astype(str).unique()) if group_col_name in df else []
    num_groups = len(unique_groups)
    if num_groups == 0: return

    ncols = 2 if num_groups > 1 else 1
    nrows = (num_groups + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.5 * ncols, 6.5 * nrows), squeeze=False,
                             facecolor=COLOR_PALETTE.get("figure_background"))
    axes = axes.flatten()

    for i, group_value in enumerate(unique_groups):
        ax = axes[i]
        ax.set_facecolor(COLOR_PALETTE.get("plot_background"))
        group_df = df[df[group_col_name] == group_value]

        if group_df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color=COLOR_PALETTE.get("text_color_dark"))
            ax.set_title(f"{plot_title_prefix}: {group_value}", color=COLOR_PALETTE.get("title_color"))
            ax.set_xticks([]); ax.set_yticks([])
            continue

        counts = group_df['vote_bin'].value_counts().reindex(labels, fill_value=0)
        percentages = (counts / counts.sum() * 100).fillna(0)

        if not counts.empty:
            bars = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, ax=ax,
                               palette="Blues_d", legend=False, dodge=False)
        else:
            ax.text(0.5, 0.5, 'No binned data', ha='center', va='center', transform=ax.transAxes, color=COLOR_PALETTE.get("text_color_dark"))

        ax.set_title(f"{plot_title_prefix}: {group_value} ({len(group_df):,} Reviews)", color=COLOR_PALETTE.get("title_color"))
        ax.set_ylabel("Number of Reviews", color=COLOR_PALETTE.get("text_color_dark"))
        ax.set_xlabel("Helpful Vote Bins", color=COLOR_PALETTE.get("text_color_dark"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", color=COLOR_PALETTE.get("text_color_dark"))
        ax.tick_params(axis='y', colors=COLOR_PALETTE.get("text_color_dark"))

        if not counts.empty and 'bars' in locals():
            for bar_idx, bar_item in enumerate(bars.patches):
                y_val = bar_item.get_height()
                percentage = percentages.iloc[bar_idx]
                text_offset = counts.max() * 0.02 if counts.max() > 0 else y_val * 0.05
                ax.text(bar_item.get_x() + bar_item.get_width()/2., y_val + text_offset,
                        f'{int(y_val):,}\n({percentage:.0f}%)', ha='center', va='bottom', fontsize=8, color=COLOR_PALETTE.get("text_color_dark"))
            ax.set_ylim(0, counts.max() * 1.20)
        ax.grid(axis='y', linestyle='--', alpha=0.7, color=COLOR_PALETTE.get("grid_color"))

    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
    plt.suptitle(f"Voting Distributions: {plot_title_prefix}", fontsize=18, y=1.03, color=COLOR_PALETTE.get("title_color"), weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, f"{output_filename_prefix}_voting_distribution.png"), facecolor=fig.get_facecolor())
    print(f"✅ Saved plot: {os.path.join(EDA_OUTPUT_DIR, f'{output_filename_prefix}_voting_distribution.png')}")
    plt.close(fig)


def plot_text_length_distribution_by_group(df, text_length_col, group_col, output_filename, title_suffix=""):
    if text_length_col not in df.columns:
        print(f"⚠️ '{text_length_col}' column not found. Skipping '{title_suffix}' text length plot.")
        return
    if group_col not in df.columns:
        print(f"⚠️ '{group_col}' column for grouping not found. Skipping '{title_suffix}' text length plot.")
        return

    # Ensure text_length_col is numeric
    df[text_length_col] = pd.to_numeric(df[text_length_col], errors='coerce')
    plot_df = df.dropna(subset=[text_length_col]).copy() # Use .copy()

    if plot_df.empty or plot_df[group_col].nunique() == 0:
        print(f"⚠️ No data or groups to plot for '{group_col}'. Skipping '{title_suffix}' text length plot.")
        return

    fig_len, ax_len = plt.subplots(figsize=(max(8, 2 * plot_df[group_col].nunique()), 7), facecolor=COLOR_PALETTE.get("figure_background"))
    ax_len.set_facecolor(COLOR_PALETTE.get("plot_background"))

    sns.boxplot(
        x=group_col,
        y=text_length_col,
        data=plot_df,
        hue=group_col, # To address FutureWarning
        ax=ax_len,
        palette=COLOR_PALETTE.get("boxplot_palette", "viridis"),
        legend=False,  # To address FutureWarning
        dodge=False,   # To address FutureWarning
        showfliers=False # <<< --- MODIFICATION: HIDE OUTLIERS FOR BETTER SCALE
    )

    title_str = f"Text Length ({text_length_col.replace('_', ' ').title()}) Distribution by {group_col.replace('_', ' ').title()}"
    if title_suffix: title_str += f" - {title_suffix}"

    ax_len.set_title(title_str, color=COLOR_PALETTE.get("title_color"), fontsize=15, weight='bold')
    ax_len.set_xlabel(group_col.replace('_', ' ').title(), color=COLOR_PALETTE.get("text_color_dark"), fontsize=12)
    ax_len.set_ylabel(text_length_col.replace('_', ' ').title(), color=COLOR_PALETTE.get("text_color_dark"), fontsize=12)
    ax_len.tick_params(axis='x', colors=COLOR_PALETTE.get("text_color_dark"))
    ax_len.tick_params(axis='y', colors=COLOR_PALETTE.get("text_color_dark"))
    plt.setp(ax_len.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Consider a more robust y-limit if showfliers=False still doesn't give good scale
    # For example, based on quantiles of the main distribution if needed, but start with showfliers=False
    # y_upper_limit = plot_df[text_length_col].quantile(0.95) # Example: up to 95th percentile
    # ax_len.set_ylim(0, y_upper_limit * 1.1)


    ax_len.grid(axis='y', linestyle='--', alpha=0.7, color=COLOR_PALETTE.get("grid_color"))
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, output_filename), facecolor=fig_len.get_facecolor())
    print(f"✅ Saved plot: {os.path.join(EDA_OUTPUT_DIR, output_filename)}")
    plt.close(fig_len)

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
        if df_raw.empty: continue
        print(f"   Loading metadata for {category}...")
        meta_df = load_and_clean_metadata(metadata_file_path)
        labeling_config_main = config.get("labeling", {})
        labeling_mode_main = labeling_config_main.get("mode", "percentile")
        print(f"   Preprocessing reviews for {category} (labeling_mode for preprocessing: {labeling_mode_main})...")
        df_processed = preprocess_reviews(df_raw.copy(), metadata_df=meta_df, labeling_mode=labeling_mode_main)
        del df_raw
        if df_processed.empty or 'helpful_vote' not in df_processed.columns: continue
        df_processed['category'] = category
        all_categories_df_list.append(df_processed)

    if not all_categories_df_list:
        print("\n❌ No data processed. EDA finished.")
        return

    combined_df = pd.concat(all_categories_df_list, ignore_index=True)
    combined_df['helpful_vote'] = pd.to_numeric(combined_df['helpful_vote'], errors='coerce').fillna(0)
    if 'year' not in combined_df.columns and 'timestamp' in combined_df.columns:
        combined_df["year"] = pd.to_datetime(combined_df["timestamp"], unit="ms", errors='coerce').dt.year
    elif 'year' not in combined_df.columns:
        print("   ⚠️ 'year' column not found. Year-based plots will be skipped.")
    print(f"\n📊 Combined DataFrame for EDA: {len(combined_df)} reviews.")

    # Plot 1: Voting distributions by category
    print("\nGenerating voting distributions by category...")
    plot_voting_distribution_generic(
        combined_df.copy(), group_col_name='category',
        plot_title_prefix='Category', output_filename_prefix='by_category'
    )

    # Plot 2: Voting distribution by price quantile
    if 'price' in combined_df.columns:
        print("\nGenerating voting distribution by price quantile...")
        combined_df['price_numeric'] = pd.to_numeric(combined_df['price'], errors='coerce')
        df_for_quantile = combined_df.dropna(subset=['price_numeric']).copy()
        if not df_for_quantile.empty and df_for_quantile['price_numeric'].nunique() >= 4:
            try:
                quantile_custom_labels = ["Q1 (Lowest Prices)", "Q2 (Low-Mid Prices)", "Q3 (Mid-High Prices)", "Q4 (Highest Prices)"]
                df_for_quantile.loc[:, 'price_quantile_label'] = pd.qcut(
                    df_for_quantile['price_numeric'], q=4, labels=quantile_custom_labels, duplicates='drop'
                )
                plot_voting_distribution_generic(
                    df_for_quantile, group_col_name='price_quantile_label',
                    plot_title_prefix='Price Quantile', output_filename_prefix='by_price_quantile'
                )
            except Exception as e:
                print(f"   ⚠️ Could not generate price quantile plot (Error: {e}). Skipping.")
        # ... (else conditions)
    # ...

    # Plot 3: Text Length Distribution by Category
    if 'review_word_count' in combined_df.columns and 'category' in combined_df.columns:
        print("\nGenerating text length (word count) distribution by category...")
        plot_text_length_distribution_by_group(
            df=combined_df.copy(), text_length_col='review_word_count', group_col='category',
            output_filename='text_length_word_count_by_category.png', title_suffix="Overall"
        )

    # --- Additional Plots from Original EDA Notebook (with styling) ---
    # Plot 4: Overall Helpful vote distribution
    # print("\nGenerating overall helpful vote distribution...")
    # fig_hist, ax_hist = plt.subplots(figsize=(10, 6), facecolor=COLOR_PALETTE.get("figure_background"))
    # ax_hist.set_facecolor(COLOR_PALETTE.get("plot_background"))
    # sns.histplot(combined_df['helpful_vote'], bins=50, log_scale=(False, True), ax=ax_hist, color=COLOR_PALETTE.get("primary_bar_color"))
    # ax_hist.set_title("Overall Distribution of Helpful Votes", color=COLOR_PALETTE.get("title_color"))
    # ax_hist.set_xlabel("Helpful Votes", color=COLOR_PALETTE.get("text_color_dark"))
    # ax_hist.set_ylabel("Frequency (log scale)", color=COLOR_PALETTE.get("text_color_dark"))
    # ax_hist.tick_params(axis='both', colors=COLOR_PALETTE.get("text_color_dark"))
    # ax_hist.grid(True, color=COLOR_PALETTE.get("grid_color"), linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.savefig(os.path.join(EDA_OUTPUT_DIR, "overall_helpful_vote_distribution.png"), facecolor=fig_hist.get_facecolor())
    # print(f"✅ Saved plot: {os.path.join(EDA_OUTPUT_DIR, 'overall_helpful_vote_distribution.png')}")
    # plt.close(fig_hist)

    if not combined_df.empty:
        zero_helpful = (combined_df['helpful_vote'] == 0).sum()
        total_reviews = len(combined_df)
        print(f"📊 Overall: Reviews with 0 helpful votes: {zero_helpful} / {total_reviews} ({zero_helpful/total_reviews:.2%})")

    # # Plot 5: Reviews per product (parent_asin) - overall
    # if 'parent_asin' in combined_df.columns and not combined_df.empty:
    #     print("\nGenerating reviews per product distribution...")
    #     asin_counts = combined_df['parent_asin'].value_counts()
    #     fig_asin, ax_asin = plt.subplots(figsize=(10, 6), facecolor=COLOR_PALETTE.get("figure_background"))
    #     ax_asin.set_facecolor(COLOR_PALETTE.get("plot_background"))
    #     sns.histplot(asin_counts, bins=50, log_scale=(True, True), ax=ax_asin, color=COLOR_PALETTE.get("primary_bar_color"))
    #     ax_asin.set_title("Distribution of Reviews per Product - Overall", color=COLOR_PALETTE.get("title_color"))
    #     ax_asin.set_xlabel("Reviews per Product (log)", color=COLOR_PALETTE.get("text_color_dark"))
    #     ax_asin.set_ylabel("Number of Products (log)", color=COLOR_PALETTE.get("text_color_dark"))
    #     ax_asin.tick_params(axis='both', colors=COLOR_PALETTE.get("text_color_dark"))
    #     ax_asin.grid(True, color=COLOR_PALETTE.get("grid_color"), linestyle='--', alpha=0.7)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(EDA_OUTPUT_DIR, "overall_reviews_per_product.png"), facecolor=fig_asin.get_facecolor())
    #     print(f"✅ Saved plot: {os.path.join(EDA_OUTPUT_DIR, 'overall_reviews_per_product.png')}")
    #     plt.close(fig_asin)

    # # Plot 6: Helpfulness vote concentration by product - overall
    # if 'parent_asin' in combined_df.columns and not combined_df.empty:
    #     print("\nGenerating helpful votes per review by product distribution...")
    #     asin_helpful_sum = combined_df.groupby('parent_asin')['helpful_vote'].sum()
    #     asin_review_counts = combined_df['parent_asin'].value_counts()
    #     concentration_df = pd.DataFrame({'review_count': asin_review_counts, 'total_helpful_votes': asin_helpful_sum}).fillna(0)
    #     concentration_df = concentration_df[concentration_df['review_count'] > 0]
    #     concentration_df['helpful_per_review'] = concentration_df['total_helpful_votes'] / concentration_df['review_count']
    #     fig_conc, ax_conc = plt.subplots(figsize=(10, 6), facecolor=COLOR_PALETTE.get("figure_background"))
    #     ax_conc.set_facecolor(COLOR_PALETTE.get("plot_background"))
    #     sns.histplot(concentration_df['helpful_per_review'], bins=50, log_scale=(False, True), ax=ax_conc, color=COLOR_PALETTE.get("primary_bar_color"))
    #     ax_conc.set_title("Distribution of Avg Helpful Votes per Review (by Product)", color=COLOR_PALETTE.get("title_color"))
    #     ax_conc.set_xlabel("Avg Helpful Votes per Review for a Product", color=COLOR_PALETTE.get("text_color_dark"))
    #     ax_conc.set_ylabel("Number of Products (log scale)", color=COLOR_PALETTE.get("text_color_dark"))
    #     ax_conc.tick_params(axis='both', colors=COLOR_PALETTE.get("text_color_dark"))
    #     ax_conc.grid(True, color=COLOR_PALETTE.get("grid_color"), linestyle='--', alpha=0.7)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(EDA_OUTPUT_DIR, "overall_helpful_per_review_by_product.png"), facecolor=fig_conc.get_facecolor())
    #     print(f"✅ Saved plot: {os.path.join(EDA_OUTPUT_DIR, 'overall_helpful_per_review_by_product.png')}")
    #     plt.close(fig_conc)

    # # Plot 7: Pareto Insight
    # if not combined_df.empty and combined_df['helpful_vote'].sum() > 0:
    #     print("\nGenerating Pareto insight for helpful votes...")
    #     sorted_votes = combined_df["helpful_vote"].sort_values(ascending=False).reset_index(drop=True)
    #     cumulative_votes = sorted_votes.cumsum()
    #     total_votes_sum = sorted_votes.sum()
    #     cumulative_share = cumulative_votes / total_votes_sum if total_votes_sum > 0 else np.zeros_like(cumulative_votes)
    #     percent_reviews = np.arange(1, len(sorted_votes) + 1) / len(sorted_votes)
    #     fig_pareto, ax_pareto = plt.subplots(figsize=(10, 6), facecolor=COLOR_PALETTE.get("figure_background"))
    #     ax_pareto.set_facecolor(COLOR_PALETTE.get("plot_background"))
    #     sns.lineplot(x=percent_reviews, y=cumulative_share, color=COLOR_PALETTE.get("line_plot_main"), linewidth=2, ax=ax_pareto)
    #     ax_pareto.axhline(0.8, color=COLOR_PALETTE.get("pareto_80_line"), linestyle="--", label="80% Helpful Votes")
    #     ax_pareto.axvline(0.2, color=COLOR_PALETTE.get("pareto_20_line"), linestyle="--", label="20% Reviews")
    #     ax_pareto.set_xlabel("Fraction of Reviews (sorted by helpfulness)", color=COLOR_PALETTE.get("text_color_dark"))
    #     ax_pareto.set_ylabel("Cumulative Share of Total Helpful Votes", color=COLOR_PALETTE.get("text_color_dark"))
    #     ax_pareto.set_title("Cumulative Distribution of Helpful Votes (Pareto Insight)", color=COLOR_PALETTE.get("title_color"))
    #     legend = ax_pareto.legend()
    #     for text in legend.get_texts(): text.set_color(COLOR_PALETTE.get("text_color_dark"))
    #     ax_pareto.tick_params(axis='both', colors=COLOR_PALETTE.get("text_color_dark"))
    #     ax_pareto.grid(True, color=COLOR_PALETTE.get("grid_color"), linestyle='--', alpha=0.7)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(EDA_OUTPUT_DIR, "overall_pareto_helpful_votes.png"), facecolor=fig_pareto.get_facecolor())
    #     print(f"✅ Saved plot: {os.path.join(EDA_OUTPUT_DIR, 'overall_pareto_helpful_votes.png')}")
    #     plt.close(fig_pareto)
    # # ... (else condition for skipping pareto)

    # Plot 8: Helpful votes per product by year - Boxplot
    if 'year' in combined_df.columns and 'parent_asin' in combined_df.columns and not combined_df.empty:
        print("\nGenerating helpful votes per product by year boxplot...")
        grouped_by_year_asin = combined_df.groupby(["year", "parent_asin"])["helpful_vote"].sum().reset_index()
        fig_box_yr, ax_box_yr = plt.subplots(figsize=(max(10, 1.5 * combined_df['year'].nunique()), 7), facecolor=COLOR_PALETTE.get("figure_background"))
        ax_box_yr.set_facecolor(COLOR_PALETTE.get("plot_background"))
        sns.boxplot(data=grouped_by_year_asin, x="year", y="helpful_vote", hue="year", # Added hue
                    showfliers=False, palette=COLOR_PALETTE.get("boxplot_palette"), legend=False, dodge=False, ax=ax_box_yr) # Added legend, dodge
        ax_box_yr.set_yscale("log")
        ax_box_yr.set_title("Distribution of Total Helpful Votes per Product, by Year", color=COLOR_PALETTE.get("title_color"))
        ax_box_yr.set_xlabel("Review Year", color=COLOR_PALETTE.get("text_color_dark"))
        ax_box_yr.set_ylabel("Total Helpful Votes per Product (log scale)", color=COLOR_PALETTE.get("text_color_dark"))
        ax_box_yr.tick_params(axis='both', colors=COLOR_PALETTE.get("text_color_dark"))
        plt.setp(ax_box_yr.get_xticklabels(), rotation=45, ha="right")
        ax_box_yr.grid(axis='y', linestyle='--', alpha=0.7, color=COLOR_PALETTE.get("grid_color"))
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_OUTPUT_DIR, "overall_helpful_votes_per_product_by_year.png"), facecolor=fig_box_yr.get_facecolor())
        print(f"✅ Saved plot: {os.path.join(EDA_OUTPUT_DIR, 'overall_helpful_votes_per_product_by_year.png')}")
        plt.close(fig_box_yr)

    # Plot 9: Total helpful votes by year - Barplot
    if 'year' in combined_df.columns and not combined_df.empty:
        print("\nGenerating total helpful votes by year barplot...")
        yearly_total_helpful_votes = combined_df.groupby("year")["helpful_vote"].sum().reset_index()
        fig_bar_yr_total, ax_bar_yr_total = plt.subplots(figsize=(max(10, 0.8 * combined_df['year'].nunique()), 6), facecolor=COLOR_PALETTE.get("figure_background"))
        ax_bar_yr_total.set_facecolor(COLOR_PALETTE.get("plot_background"))
        sns.barplot(data=yearly_total_helpful_votes, x="year", y="helpful_vote", hue="year", # Added hue
                    palette=COLOR_PALETTE.get("barplot_palette_year"), legend=False, dodge=False, ax=ax_bar_yr_total) # Added legend, dodge
        ax_bar_yr_total.set_yscale("log")
        ax_bar_yr_total.set_title("Total Helpful Votes by Review Year - Overall", color=COLOR_PALETTE.get("title_color"))
        ax_bar_yr_total.set_ylabel("Total Helpful Votes (log scale)", color=COLOR_PALETTE.get("text_color_dark"))
        ax_bar_yr_total.set_xlabel("Review Year", color=COLOR_PALETTE.get("text_color_dark"))
        ax_bar_yr_total.tick_params(axis='both', colors=COLOR_PALETTE.get("text_color_dark"))
        plt.setp(ax_bar_yr_total.get_xticklabels(), rotation=45, ha="right")
        ax_bar_yr_total.grid(axis='y', linestyle='--', alpha=0.7, color=COLOR_PALETTE.get("grid_color"))
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_OUTPUT_DIR, "overall_total_helpful_votes_by_year.png"), facecolor=fig_bar_yr_total.get_facecolor())
        print(f"✅ Saved plot: {os.path.join(EDA_OUTPUT_DIR, 'overall_total_helpful_votes_by_year.png')}")
        plt.close(fig_bar_yr_total)

    print("\n--- EDA Script Finished ---")

if __name__ == "__main__":
    perform_eda()