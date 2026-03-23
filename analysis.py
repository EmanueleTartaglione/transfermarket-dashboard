"""
Premier League Player Market Value Analysis
Explores what drives player valuations using data scraped from Transfermarkt.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# Style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

OUTPUT_DIR = "plots"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_clean(path="premier_league_players.csv"):
    df = pd.read_csv(path)
    # Drop players without market value
    df = df.dropna(subset=["market_value_eur"])
    # Parse height to float
    df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")
    # Market value in millions for readability
    df["value_m"] = df["market_value_eur"] / 1_000_000
    # Contract years remaining (approximate from contract_expires)
    df["contract_expires"] = pd.to_datetime(df["contract_expires"], format="%d/%m/%Y", errors="coerce")
    today = pd.Timestamp("2026-03-21")
    df["contract_years_left"] = ((df["contract_expires"] - today).dt.days / 365.25).round(2)
    df["contract_years_left"] = df["contract_years_left"].clip(lower=0)
    # Simplified position groups
    pos_map = {
        "Goalkeeper": "Goalkeeper",
        "Centre-Back": "Defender",
        "Left-Back": "Defender",
        "Right-Back": "Defender",
        "Defensive Midfield": "Midfielder",
        "Central Midfield": "Midfielder",
        "Attacking Midfield": "Midfielder",
        "Right Midfield": "Midfielder",
        "Left Midfield": "Midfielder",
        "Left Winger": "Forward",
        "Right Winger": "Forward",
        "Centre-Forward": "Forward",
        "Second Striker": "Forward",
    }
    df["position_group"] = df["position"].map(pos_map).fillna("Other")
    # Primary nationality (first listed)
    df["primary_nationality"] = df["nationality"].str.split(",").str[0].str.strip()
    return df


# ─────────────────────────────────────────────
# 1. VALUE DISTRIBUTION
# ─────────────────────────────────────────────
def plot_value_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram (log scale)
    axes[0].hist(df["value_m"], bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Market Value (€M)")
    axes[0].set_ylabel("Number of Players")
    axes[0].set_title("Distribution of Player Market Values")
    axes[0].axvline(df["value_m"].median(), color="red", ls="--", label=f'Median: €{df["value_m"].median():.1f}M')
    axes[0].axvline(df["value_m"].mean(), color="orange", ls="--", label=f'Mean: €{df["value_m"].mean():.1f}M')
    axes[0].legend()

    # Box plot by position group
    order = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
    sns.boxplot(data=df, x="position_group", y="value_m", order=order, ax=axes[1], palette="Set2")
    axes[1].set_xlabel("Position Group")
    axes[1].set_ylabel("Market Value (€M)")
    axes[1].set_title("Market Value by Position Group")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_value_distribution.png", bbox_inches="tight")
    plt.close()
    print("  [1] Value distribution saved")


# ─────────────────────────────────────────────
# 2. AGE vs VALUE (with position breakdown)
# ─────────────────────────────────────────────
def plot_age_value(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    groups = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
    for g, c in zip(groups, colors):
        sub = df[df["position_group"] == g]
        axes[0].scatter(sub["age"], sub["value_m"], alpha=0.45, s=30, label=g, color=c)
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Market Value (€M)")
    axes[0].set_title("Age vs Market Value")
    axes[0].legend()

    # Average value by age and position
    for g, c in zip(groups, colors):
        sub = df[df["position_group"] == g]
        avg = sub.groupby("age")["value_m"].mean()
        if len(avg) > 3:
            axes[1].plot(avg.index, avg.values, marker="o", markersize=4, label=g, color=c, linewidth=2)
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Average Market Value (€M)")
    axes[1].set_title("Average Value by Age & Position")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_age_vs_value.png", bbox_inches="tight")
    plt.close()
    print("  [2] Age vs value saved")


# ─────────────────────────────────────────────
# 3. CLUB SPENDING POWER
# ─────────────────────────────────────────────
def plot_club_values(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Total squad value
    club_total = df.groupby("club")["value_m"].sum().sort_values(ascending=True)
    club_total.plot.barh(ax=axes[0], color="#4C72B0")
    axes[0].set_xlabel("Total Squad Value (€M)")
    axes[0].set_title("Total Squad Market Value by Club")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}M"))

    # Average player value
    club_avg = df.groupby("club")["value_m"].mean().sort_values(ascending=True)
    club_avg.plot.barh(ax=axes[1], color="#E67E22")
    axes[1].set_xlabel("Average Player Value (€M)")
    axes[1].set_title("Average Player Value by Club")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}M"))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_club_values.png", bbox_inches="tight")
    plt.close()
    print("  [3] Club values saved")


# ─────────────────────────────────────────────
# 4. NATIONALITY ANALYSIS
# ─────────────────────────────────────────────
def plot_nationality(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top 15 nationalities by count
    nat_count = df["primary_nationality"].value_counts().head(15)
    nat_count.plot.barh(ax=axes[0], color="#2ECC71")
    axes[0].set_xlabel("Number of Players")
    axes[0].set_title("Top 15 Nationalities (Player Count)")
    axes[0].invert_yaxis()

    # Top 15 nationalities by average value (min 5 players)
    nat_groups = df.groupby("primary_nationality")
    nat_avg = nat_groups["value_m"].mean()
    nat_sizes = nat_groups.size()
    qualified = nat_avg[nat_sizes >= 5].sort_values(ascending=False).head(15)
    qualified.sort_values(ascending=True).plot.barh(ax=axes[1], color="#9B59B6")
    axes[1].set_xlabel("Average Market Value (€M)")
    axes[1].set_title("Avg Value by Nationality (min 5 players)")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}M"))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_nationality.png", bbox_inches="tight")
    plt.close()
    print("  [4] Nationality analysis saved")


# ─────────────────────────────────────────────
# 5. HEIGHT & FOOT ANALYSIS
# ─────────────────────────────────────────────
def plot_physical(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Height vs value by position
    sub = df.dropna(subset=["height_m"])
    groups = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
    for g, c in zip(groups, colors):
        s = sub[sub["position_group"] == g]
        axes[0].scatter(s["height_m"], s["value_m"], alpha=0.4, s=30, label=g, color=c)
    axes[0].set_xlabel("Height (m)")
    axes[0].set_ylabel("Market Value (€M)")
    axes[0].set_title("Height vs Market Value")
    axes[0].legend()

    # Preferred foot
    foot_data = df[df["foot"].isin(["right", "left", "both"])]
    foot_stats = foot_data.groupby("foot").agg(
        count=("value_m", "size"),
        avg_value=("value_m", "mean"),
        median_value=("value_m", "median"),
    ).reset_index()

    x = range(len(foot_stats))
    width = 0.35
    bars1 = axes[1].bar([i - width/2 for i in x], foot_stats["avg_value"], width, label="Mean", color="#3498DB")
    bars2 = axes[1].bar([i + width/2 for i in x], foot_stats["median_value"], width, label="Median", color="#E67E22")
    axes[1].set_xticks(list(x))
    labels = [f'{row["foot"].title()}\n(n={row["count"]})' for _, row in foot_stats.iterrows()]
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Market Value (€M)")
    axes[1].set_title("Market Value by Preferred Foot")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_physical.png", bbox_inches="tight")
    plt.close()
    print("  [5] Physical analysis saved")


# ─────────────────────────────────────────────
# 6. CONTRACT LENGTH vs VALUE
# ─────────────────────────────────────────────
def plot_contract(df):
    fig, ax = plt.subplots(figsize=(10, 5))

    sub = df.dropna(subset=["contract_years_left"])
    sub = sub[sub["contract_years_left"] <= 8]

    ax.scatter(sub["contract_years_left"], sub["value_m"], alpha=0.4, s=25, color="#4C72B0")
    # Trend line
    z = np.polyfit(sub["contract_years_left"], sub["value_m"], 2)
    p = np.poly1d(z)
    x_line = np.linspace(0, sub["contract_years_left"].max(), 100)
    ax.plot(x_line, p(x_line), color="red", linewidth=2, label="Trend (quadratic)")

    ax.set_xlabel("Contract Years Remaining")
    ax.set_ylabel("Market Value (€M)")
    ax.set_title("Contract Length vs Market Value")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_contract_value.png", bbox_inches="tight")
    plt.close()
    print("  [6] Contract analysis saved")


# ─────────────────────────────────────────────
# 7. CORRELATION HEATMAP
# ─────────────────────────────────────────────
def plot_correlations(df):
    fig, ax = plt.subplots(figsize=(8, 6))

    numeric_cols = ["age", "height_m", "value_m", "contract_years_left"]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f",
                square=True, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_correlations.png", bbox_inches="tight")
    plt.close()
    print("  [7] Correlation heatmap saved")


# ─────────────────────────────────────────────
# 8. PREDICTIVE MODEL
# ─────────────────────────────────────────────
def build_model(df):
    print("\n  Building predictive model...")

    features_df = df.dropna(subset=["age", "height_m", "contract_years_left"]).copy()

    # Encode categoricals
    le_pos = LabelEncoder()
    features_df["position_enc"] = le_pos.fit_transform(features_df["position_group"])
    le_foot = LabelEncoder()
    features_df["foot_enc"] = le_foot.fit_transform(features_df["foot"].fillna("unknown"))
    le_club = LabelEncoder()
    features_df["club_enc"] = le_club.fit_transform(features_df["club"])

    feature_cols = ["age", "height_m", "contract_years_left", "position_enc", "foot_enc", "club_enc"]
    X = features_df[feature_cols]
    y = features_df["value_m"]

    # Compare models with cross-validation
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
    }

    print("\n  Cross-validation R² scores (5-fold):")
    best_name, best_score, best_model = None, -999, None
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        mae_scores = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
        print(f"    {name:25s}  R²: {scores.mean():.3f} ± {scores.std():.3f}  |  MAE: €{mae_scores.mean():.2f}M")
        if scores.mean() > best_score:
            best_name, best_score, best_model = name, scores.mean(), model

    # Fit best model on full data for feature importance & residuals
    best_model.fit(X, y)
    y_pred = best_model.predict(X)

    # Feature importance (for tree models)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_cols)
        importances.sort_values().plot.barh(ax=axes[0], color="#4C72B0")
        axes[0].set_title(f"Feature Importance ({best_name})")
        axes[0].set_xlabel("Importance")

    # Actual vs Predicted
    axes[1].scatter(y, y_pred, alpha=0.4, s=20, color="#4C72B0")
    max_val = max(y.max(), y_pred.max())
    axes[1].plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")
    axes[1].set_xlabel("Actual Value (€M)")
    axes[1].set_ylabel("Predicted Value (€M)")
    axes[1].set_title(f"Actual vs Predicted ({best_name})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_model.png", bbox_inches="tight")
    plt.close()
    print(f"  [8] Model plots saved (best: {best_name})")

    # Over/undervalued players
    features_df["predicted_value"] = y_pred
    features_df["residual"] = features_df["value_m"] - features_df["predicted_value"]

    print("\n  Top 10 OVERVALUED players (actual >> predicted):")
    over = features_df.nlargest(10, "residual")[["name", "club", "position", "age", "value_m", "predicted_value", "residual"]]
    over = over.copy()
    over["predicted_value"] = over["predicted_value"].round(1)
    over["residual"] = over["residual"].round(1)
    print(over.to_string(index=False))

    print("\n  Top 10 UNDERVALUED players (actual << predicted):")
    under = features_df.nsmallest(10, "residual")[["name", "club", "position", "age", "value_m", "predicted_value", "residual"]]
    under = under.copy()
    under["predicted_value"] = under["predicted_value"].round(1)
    under["residual"] = under["residual"].round(1)
    print(under.to_string(index=False))

    # Save over/undervalued to CSV
    features_df[["name", "club", "position", "age", "value_m", "predicted_value", "residual"]].to_csv(
        "player_valuations.csv", index=False
    )
    print("\n  Saved player_valuations.csv with predicted values & residuals")


# ─────────────────────────────────────────────
# 9. POSITION DETAILED BREAKDOWN
# ─────────────────────────────────────────────
def plot_position_detail(df):
    fig, ax = plt.subplots(figsize=(14, 6))

    pos_order = df.groupby("position")["value_m"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="position", y="value_m", order=pos_order, palette="husl", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Position")
    ax.set_ylabel("Market Value (€M)")
    ax.set_title("Market Value Distribution by Detailed Position")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/09_position_detail.png", bbox_inches="tight")
    plt.close()
    print("  [9] Detailed position breakdown saved")


# ─────────────────────────────────────────────
# 10. AGE PROFILE BY CLUB
# ─────────────────────────────────────────────
def plot_club_age_profile(df):
    fig, ax = plt.subplots(figsize=(14, 6))

    club_order = df.groupby("club")["value_m"].sum().sort_values(ascending=False).index
    sns.boxplot(data=df, x="club", y="age", order=club_order, palette="coolwarm", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Club")
    ax.set_ylabel("Age")
    ax.set_title("Squad Age Profile by Club (ordered by total squad value)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_club_age_profile.png", bbox_inches="tight")
    plt.close()
    print("  [10] Club age profile saved")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Premier League Market Value Analysis")
    print("=" * 60)

    df = load_and_clean()
    print(f"\nLoaded {len(df)} players with market values")
    print(f"Value range: €{df['value_m'].min():.2f}M - €{df['value_m'].max():.1f}M")
    print(f"Mean: €{df['value_m'].mean():.1f}M  |  Median: €{df['value_m'].median():.1f}M")

    print("\nGenerating plots...")
    plot_value_distribution(df)
    plot_age_value(df)
    plot_club_values(df)
    plot_nationality(df)
    plot_physical(df)
    plot_contract(df)
    plot_correlations(df)
    plot_position_detail(df)
    plot_club_age_profile(df)
    build_model(df)

    print("\n" + "=" * 60)
    print(f"All plots saved to '{OUTPUT_DIR}/' folder")
    print("=" * 60)


if __name__ == "__main__":
    main()
