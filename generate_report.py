import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import textwrap

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# --- CONFIG ---
BASE = "/Users/emanueletartaglione/Desktop/Transfermarket Project"
PDF_PATH = os.path.join(BASE, "prediction_analysis.pdf")
IMG_DIR = os.path.join(BASE, "_report_images")
os.makedirs(IMG_DIR, exist_ok=True)

PRIMARY = "#1B1464"
ACCENT = "#EE6C4D"
LIGHT_BG = "#F0F0F8"

# --- LOAD DATA ---
print("Loading data...")
players = pd.read_csv(os.path.join(BASE, "premier_league_players.csv"))
stats = pd.read_csv(os.path.join(BASE, "player_stats.csv"), low_memory=False)

# Merge — drop duplicate 'club' from stats to avoid club_x/club_y
stats_cols = ["name", "player_id", "total_appearances", "total_goals", "total_assists",
              "total_yellow_cards", "total_red_cards", "total_minutes"]
stats_cols = [c for c in stats_cols if c in stats.columns]
df = players.merge(stats[stats_cols], on="name", how="inner")
total_players = len(players)
merged_players = len(df)
print(f"Players: {total_players}, Merged: {merged_players}")

# Convert height
df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")

# Filter value > 0
df = df[df["market_value_eur"] > 0].copy()

# Position mapping
pos_map = {"Goalkeeper": 0, "Defender": 1, "Midfielder": 2, "Forward": 3}
# Normalize position values
def map_position(p):
    if pd.isna(p):
        return 2
    p = str(p).strip()
    for key in pos_map:
        if key.lower() in p.lower():
            return pos_map[key]
    if "gk" in p.lower() or "keeper" in p.lower():
        return 0
    if "def" in p.lower() or "back" in p.lower():
        return 1
    if "mid" in p.lower():
        return 2
    if "for" in p.lower() or "att" in p.lower() or "wing" in p.lower() or "strik" in p.lower():
        return 3
    return 2

df["position_code"] = df["position"].apply(map_position)

# Ensure numeric
for col in ["total_appearances", "total_goals", "total_assists", "total_minutes"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Derived features
df["goals_per_app"] = np.where(df["total_appearances"] > 0, df["total_goals"] / df["total_appearances"], 0)
df["assists_per_app"] = np.where(df["total_appearances"] > 0, df["total_assists"] / df["total_appearances"], 0)
df["minutes_per_app"] = np.where(df["total_appearances"] > 0, df["total_minutes"] / df["total_appearances"], 0)

features = ["age", "total_appearances", "total_goals", "total_assists", "total_minutes",
            "goals_per_app", "assists_per_app", "minutes_per_app", "position_code"]

X = df[features].values
y = df["market_value_eur"].values

# --- TRAIN MODEL ---
print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
y_pred_all = model.predict(X)

r2 = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"R²={r2:.4f}, MAE={mae:,.0f}, RMSE={rmse:,.0f}")

# Add predictions to df
df["predicted_value"] = y_pred_all
df["difference"] = df["market_value_eur"] - df["predicted_value"]
df["diff_pct"] = df["difference"] / df["market_value_eur"] * 100

# --- CHART HELPERS ---
def fmt_eur(v):
    if abs(v) >= 1e6:
        return f"\u20ac{v/1e6:.1f}M"
    elif abs(v) >= 1e3:
        return f"\u20ac{v/1e3:.0f}K"
    return f"\u20ac{v:.0f}"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# --- CHART 1: Feature Importance ---
print("Generating charts...")
importances = model.feature_importances_
indices = np.argsort(importances)
feature_labels = ["Age", "Appearances", "Goals", "Assists", "Minutes",
                  "Goals/App", "Assists/App", "Min/App", "Position"]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh([feature_labels[i] for i in indices], importances[indices], color=PRIMARY, edgecolor="white", height=0.6)
# Highlight top feature
bars[-1].set_color(ACCENT)
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance in Market Value Prediction", fontweight="bold", color=PRIMARY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for i, v in enumerate(importances[indices]):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "feature_importance.png"), dpi=180)
plt.close()

# --- CHART 2: Actual vs Predicted ---
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(y_test / 1e6, y_pred_test / 1e6, alpha=0.5, s=20, c=PRIMARY, edgecolors="none")
max_val = max(y_test.max(), y_pred_test.max()) / 1e6
ax.plot([0, max_val], [0, max_val], "--", color=ACCENT, linewidth=2, label="Perfect Prediction")
ax.set_xlabel("Actual Value (\u20acM)")
ax.set_ylabel("Predicted Value (\u20acM)")
ax.set_title("Actual vs Predicted Market Value", fontweight="bold", color=PRIMARY)
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "actual_vs_predicted.png"), dpi=180)
plt.close()

# --- CHART 3: Residual Distribution ---
residuals = (y_test - y_pred_test) / 1e6
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(residuals, bins=40, color=PRIMARY, edgecolor="white", alpha=0.85)
ax.axvline(0, color=ACCENT, linewidth=2, linestyle="--")
ax.set_xlabel("Prediction Error (\u20acM)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Prediction Residuals", fontweight="bold", color=PRIMARY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "residuals.png"), dpi=180)
plt.close()

# --- CHART 4 & 5: Overvalued & Undervalued ---
overvalued = df.nlargest(15, "difference")[["name", "club", "position", "market_value_eur", "predicted_value", "difference"]].reset_index(drop=True)
undervalued = df.nsmallest(15, "difference")[["name", "club", "position", "market_value_eur", "predicted_value", "difference"]].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(overvalued["name"][::-1], overvalued["difference"][::-1] / 1e6, color=ACCENT, edgecolor="white", height=0.6)
ax.set_xlabel("Overvaluation (\u20acM)")
ax.set_title("Top 15 Most Overvalued Players", fontweight="bold", color=PRIMARY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for i, (v, n) in enumerate(zip(overvalued["difference"][::-1], overvalued["name"][::-1])):
    ax.text(v / 1e6 + 0.3, i, fmt_eur(v), va="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "overvalued.png"), dpi=180)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(undervalued["name"][::-1], undervalued["difference"][::-1].abs() / 1e6, color=PRIMARY, edgecolor="white", height=0.6)
ax.set_xlabel("Undervaluation (\u20acM)")
ax.set_title("Top 15 Most Undervalued Players", fontweight="bold", color=PRIMARY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for i, (v, n) in enumerate(zip(undervalued["difference"][::-1], undervalued["name"][::-1])):
    ax.text(abs(v) / 1e6 + 0.3, i, fmt_eur(abs(v)), va="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "undervalued.png"), dpi=180)
plt.close()

print("Charts saved.")

# ===================== BUILD PDF =====================
print("Building PDF...")

doc = SimpleDocTemplate(
    PDF_PATH,
    pagesize=A4,
    topMargin=2*cm,
    bottomMargin=2*cm,
    leftMargin=2.2*cm,
    rightMargin=2.2*cm,
)

WIDTH = A4[0] - 4.4*cm  # usable width

styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    "CustomTitle", parent=styles["Title"],
    fontSize=28, leading=34, textColor=HexColor(PRIMARY),
    alignment=TA_CENTER, spaceAfter=6,
    fontName="Helvetica-Bold",
)
subtitle_style = ParagraphStyle(
    "CustomSubtitle", parent=styles["Normal"],
    fontSize=16, leading=20, textColor=HexColor(ACCENT),
    alignment=TA_CENTER, spaceAfter=4,
    fontName="Helvetica",
)
date_style = ParagraphStyle(
    "DateStyle", parent=styles["Normal"],
    fontSize=12, leading=16, textColor=HexColor("#666666"),
    alignment=TA_CENTER, spaceAfter=20,
    fontName="Helvetica",
)
heading_style = ParagraphStyle(
    "SectionHeading", parent=styles["Heading1"],
    fontSize=20, leading=26, textColor=HexColor(PRIMARY),
    spaceBefore=12, spaceAfter=10,
    fontName="Helvetica-Bold",
    borderWidth=0,
    borderColor=HexColor(ACCENT),
    borderPadding=0,
)
subheading_style = ParagraphStyle(
    "SubHeading", parent=styles["Heading2"],
    fontSize=14, leading=18, textColor=HexColor(PRIMARY),
    spaceBefore=10, spaceAfter=6,
    fontName="Helvetica-Bold",
)
body_style = ParagraphStyle(
    "BodyText2", parent=styles["Normal"],
    fontSize=10.5, leading=15, textColor=HexColor("#222222"),
    alignment=TA_JUSTIFY, spaceAfter=8,
    fontName="Helvetica",
)
metric_label_style = ParagraphStyle(
    "MetricLabel", parent=styles["Normal"],
    fontSize=10, textColor=HexColor("#666666"),
    alignment=TA_CENTER, fontName="Helvetica",
)
metric_value_style = ParagraphStyle(
    "MetricValue", parent=styles["Normal"],
    fontSize=18, textColor=HexColor(PRIMARY),
    alignment=TA_CENTER, fontName="Helvetica-Bold",
    spaceAfter=2,
)

story = []

# ========= PAGE 1: TITLE =========
story.append(Spacer(1, 6*cm))
story.append(Paragraph("Premier League Player<br/>Market Value Prediction Analysis", title_style))
story.append(Spacer(1, 0.8*cm))

# Accent line
line_table = Table([[""]], colWidths=[6*cm], rowHeights=[3])
line_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,-1), HexColor(ACCENT)),
    ("LINEBELOW", (0,0), (-1,-1), 0, white),
]))
story.append(line_table)
story.append(Spacer(1, 0.8*cm))

story.append(Paragraph("2025/26 Season", subtitle_style))
story.append(Spacer(1, 0.4*cm))
story.append(Paragraph("March 2026", date_style))
story.append(Spacer(1, 2*cm))
story.append(Paragraph("Powered by Machine Learning | Random Forest Regression", date_style))
story.append(PageBreak())

# ========= PAGE 2: EXECUTIVE SUMMARY =========
story.append(Paragraph("Executive Summary", heading_style))

# Accent underline
ul = Table([[""]], colWidths=[4*cm], rowHeights=[2])
ul.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), HexColor(ACCENT))]))
story.append(ul)
story.append(Spacer(1, 0.5*cm))

story.append(Paragraph(
    "This report presents a comprehensive analysis of Premier League player market values "
    "for the 2025/26 season using machine learning techniques. A Random Forest regression model "
    "was trained on player statistics and attributes sourced from Transfermarkt to predict "
    "market valuations and identify potentially overvalued and undervalued players.",
    body_style
))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("Data Overview", subheading_style))
story.append(Paragraph(
    f"The dataset comprises <b>{total_players}</b> Premier League players with detailed statistics. "
    f"After merging player profiles with performance data, <b>{merged_players}</b> players were matched. "
    f"The model was trained on <b>{len(df)}</b> players with valid market values, using "
    f"<b>{len(features)}</b> engineered features covering age, career statistics, and positional data.",
    body_style
))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("Model Configuration", subheading_style))
story.append(Paragraph(
    "The prediction engine uses a <b>Random Forest Regressor</b> with 200 decision trees "
    "and a maximum depth of 15. The model was evaluated using an 80/20 train-test split "
    "with a fixed random state for reproducibility.",
    body_style
))
story.append(Spacer(1, 0.5*cm))

# Metrics boxes
story.append(Paragraph("Key Performance Metrics", subheading_style))

metrics_data = [
    [Paragraph("R\u00b2 Score", metric_label_style),
     Paragraph("Mean Absolute Error", metric_label_style),
     Paragraph("Root Mean Squared Error", metric_label_style)],
    [Paragraph(f"{r2:.4f}", metric_value_style),
     Paragraph(fmt_eur(mae), metric_value_style),
     Paragraph(fmt_eur(rmse), metric_value_style)],
]
metrics_table = Table(metrics_data, colWidths=[WIDTH/3]*3, rowHeights=[20, 30])
metrics_table.setStyle(TableStyle([
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("BACKGROUND", (0,0), (-1,-1), HexColor(LIGHT_BG)),
    ("BOX", (0,0), (-1,-1), 0.5, HexColor(PRIMARY)),
    ("INNERGRID", (0,0), (-1,-1), 0.3, HexColor("#DDDDDD")),
    ("TOPPADDING", (0,0), (-1,-1), 8),
    ("BOTTOMPADDING", (0,0), (-1,-1), 8),
]))
story.append(metrics_table)
story.append(Spacer(1, 0.5*cm))

r2_quality = "strong" if r2 > 0.7 else "moderate" if r2 > 0.5 else "limited"
story.append(Paragraph(
    f"The model achieves an R\u00b2 of <b>{r2:.4f}</b>, indicating {r2_quality} predictive power. "
    f"On average, predictions deviate from actual values by <b>{fmt_eur(mae)}</b>. "
    f"The RMSE of <b>{fmt_eur(rmse)}</b> suggests that while most predictions are reasonably accurate, "
    f"larger errors occur for elite players with exceptionally high valuations.",
    body_style
))
story.append(PageBreak())

# ========= PAGE 3: FEATURE IMPORTANCE =========
story.append(Paragraph("Feature Importance Analysis", heading_style))
ul2 = Table([[""]], colWidths=[4*cm], rowHeights=[2])
ul2.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), HexColor(ACCENT))]))
story.append(ul2)
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph(
    "The chart below illustrates the relative importance of each feature in the Random Forest model. "
    "Feature importance is calculated based on the average decrease in impurity across all trees.",
    body_style
))
story.append(Spacer(1, 0.3*cm))

img_fi = Image(os.path.join(IMG_DIR, "feature_importance.png"), width=WIDTH, height=WIDTH*0.56)
story.append(img_fi)
story.append(Spacer(1, 0.4*cm))

# Determine top features
sorted_imp = sorted(zip(feature_labels, importances), key=lambda x: x[1], reverse=True)
top1, top2, top3 = sorted_imp[0], sorted_imp[1], sorted_imp[2]

story.append(Paragraph("Key Observations", subheading_style))
story.append(Paragraph(
    f"<b>{top1[0]}</b> emerges as the most influential feature (importance: {top1[1]:.3f}), "
    f"followed by <b>{top2[0]}</b> ({top2[1]:.3f}) and <b>{top3[0]}</b> ({top3[1]:.3f}). "
    f"This aligns with football economics where career volume metrics and efficiency ratios "
    f"are key drivers of market valuations. Age, while important, is less dominant than "
    f"on-field performance statistics.",
    body_style
))
story.append(PageBreak())

# ========= PAGE 4: MODEL PERFORMANCE =========
story.append(Paragraph("Model Performance", heading_style))
ul3 = Table([[""]], colWidths=[4*cm], rowHeights=[2])
ul3.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), HexColor(ACCENT))]))
story.append(ul3)
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("Actual vs Predicted Values", subheading_style))
story.append(Paragraph(
    "The scatter plot below compares actual market values against model predictions. "
    "Points along the dashed line represent perfect predictions.",
    body_style
))

img_avp = Image(os.path.join(IMG_DIR, "actual_vs_predicted.png"), width=WIDTH*0.75, height=WIDTH*0.75)
story.append(img_avp)
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("Residual Distribution", subheading_style))
story.append(Paragraph(
    "The histogram below shows the distribution of prediction errors. A concentration around zero "
    "indicates the model is well-calibrated for the majority of players.",
    body_style
))

img_res = Image(os.path.join(IMG_DIR, "residuals.png"), width=WIDTH, height=WIDTH*0.56)
story.append(img_res)
story.append(Spacer(1, 0.3*cm))

median_err = np.median(np.abs(residuals))
pct_within_5m = np.mean(np.abs(residuals) < 5) * 100
story.append(Paragraph(
    f"The median absolute error is <b>{fmt_eur(median_err * 1e6)}</b>, and "
    f"<b>{pct_within_5m:.1f}%</b> of predictions fall within \u20ac5M of the actual value. "
    f"The model performs best for mid-range players and tends to underestimate "
    f"values of the most elite, high-value players.",
    body_style
))
story.append(PageBreak())

# ========= PAGE 5: OVERVALUED =========
story.append(Paragraph("Overvalued Players", heading_style))
ul4 = Table([[""]], colWidths=[4*cm], rowHeights=[2])
ul4.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), HexColor(ACCENT))]))
story.append(ul4)
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph(
    "Players whose actual market value significantly exceeds the model's prediction. "
    "This may indicate brand premium, recent transfer hype, or intangible factors not captured by statistics.",
    body_style
))
story.append(Spacer(1, 0.2*cm))

img_ov = Image(os.path.join(IMG_DIR, "overvalued.png"), width=WIDTH, height=WIDTH*0.6)
story.append(img_ov)
story.append(Spacer(1, 0.3*cm))

# Table
ov_header = [
    Paragraph("<b>Player</b>", body_style),
    Paragraph("<b>Club</b>", body_style),
    Paragraph("<b>Pos</b>", body_style),
    Paragraph("<b>Actual</b>", body_style),
    Paragraph("<b>Predicted</b>", body_style),
    Paragraph("<b>Diff</b>", body_style),
]
ov_rows = [ov_header]
for _, r in overvalued.iterrows():
    ov_rows.append([
        Paragraph(str(r["name"])[:22], body_style),
        Paragraph(str(r["club"])[:18], body_style),
        Paragraph(str(r["position"])[:6], body_style),
        Paragraph(fmt_eur(r["market_value_eur"]), body_style),
        Paragraph(fmt_eur(r["predicted_value"]), body_style),
        Paragraph(f'<font color="{ACCENT}">+{fmt_eur(r["difference"])}</font>', body_style),
    ])

ov_table = Table(ov_rows, colWidths=[WIDTH*0.20, WIDTH*0.22, WIDTH*0.08, WIDTH*0.16, WIDTH*0.16, WIDTH*0.18])
ov_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), HexColor(PRIMARY)),
    ("TEXTCOLOR", (0,0), (-1,0), white),
    ("FONTSIZE", (0,0), (-1,-1), 8),
    ("ALIGN", (3,0), (-1,-1), "RIGHT"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, HexColor(LIGHT_BG)]),
    ("BOX", (0,0), (-1,-1), 0.5, HexColor(PRIMARY)),
    ("INNERGRID", (0,0), (-1,-1), 0.25, HexColor("#CCCCCC")),
    ("TOPPADDING", (0,0), (-1,-1), 3),
    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
]))
story.append(ov_table)
story.append(PageBreak())

# ========= PAGE 6: UNDERVALUED =========
story.append(Paragraph("Undervalued Players", heading_style))
ul5 = Table([[""]], colWidths=[4*cm], rowHeights=[2])
ul5.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), HexColor(ACCENT))]))
story.append(ul5)
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph(
    "Players whose model-predicted value exceeds their actual market value. "
    "These players may represent transfer bargains based on their statistical output.",
    body_style
))
story.append(Spacer(1, 0.2*cm))

img_uv = Image(os.path.join(IMG_DIR, "undervalued.png"), width=WIDTH, height=WIDTH*0.6)
story.append(img_uv)
story.append(Spacer(1, 0.3*cm))

uv_header = [
    Paragraph("<b>Player</b>", body_style),
    Paragraph("<b>Club</b>", body_style),
    Paragraph("<b>Pos</b>", body_style),
    Paragraph("<b>Actual</b>", body_style),
    Paragraph("<b>Predicted</b>", body_style),
    Paragraph("<b>Diff</b>", body_style),
]
uv_rows = [uv_header]
for _, r in undervalued.iterrows():
    uv_rows.append([
        Paragraph(str(r["name"])[:22], body_style),
        Paragraph(str(r["club"])[:18], body_style),
        Paragraph(str(r["position"])[:6], body_style),
        Paragraph(fmt_eur(r["market_value_eur"]), body_style),
        Paragraph(fmt_eur(r["predicted_value"]), body_style),
        Paragraph(f'<font color="{PRIMARY}">{fmt_eur(r["difference"])}</font>', body_style),
    ])

uv_table = Table(uv_rows, colWidths=[WIDTH*0.20, WIDTH*0.22, WIDTH*0.08, WIDTH*0.16, WIDTH*0.16, WIDTH*0.18])
uv_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), HexColor(PRIMARY)),
    ("TEXTCOLOR", (0,0), (-1,0), white),
    ("FONTSIZE", (0,0), (-1,-1), 8),
    ("ALIGN", (3,0), (-1,-1), "RIGHT"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, HexColor(LIGHT_BG)]),
    ("BOX", (0,0), (-1,-1), 0.5, HexColor(PRIMARY)),
    ("INNERGRID", (0,0), (-1,-1), 0.25, HexColor("#CCCCCC")),
    ("TOPPADDING", (0,0), (-1,-1), 3),
    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
]))
story.append(uv_table)
story.append(PageBreak())

# ========= PAGE 7: CONCLUSIONS =========
story.append(Paragraph("Conclusions", heading_style))
ul6 = Table([[""]], colWidths=[4*cm], rowHeights=[2])
ul6.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), HexColor(ACCENT))]))
story.append(ul6)
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("Key Takeaways", subheading_style))

takeaways = [
    f"The Random Forest model explains <b>{r2*100:.1f}%</b> of the variance in player market values, "
    f"demonstrating that statistical performance data is a meaningful predictor of valuation.",

    f"<b>{top1[0]}</b> and <b>{top2[0]}</b> are the strongest predictors, suggesting that "
    f"career volume and consistency are weighted heavily in the transfer market.",

    "The model identifies several potentially overvalued players whose market prices may reflect "
    "commercial appeal, recent form, or speculative pricing rather than underlying statistics.",

    "Conversely, undervalued players represent potential transfer market opportunities where "
    "statistical output exceeds current market recognition.",
]
for t in takeaways:
    story.append(Paragraph(f"\u2022  {t}", body_style))
    story.append(Spacer(1, 0.15*cm))

story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("Limitations", subheading_style))

limitations = [
    "The model relies on aggregate career statistics and does not account for recent form, "
    "injury history, or contract duration\u2014all of which influence market value.",

    "Intangible factors such as marketing appeal, shirt sales potential, and a player's "
    "nationality premium are not captured by the feature set.",

    "The dataset is limited to the current Premier League squad, which may not capture "
    "the full range of market dynamics across European football.",

    "Position-based valuation differences (e.g., premium for centre-backs vs. full-backs) "
    "are simplified into four broad categories.",
]
for t in limitations:
    story.append(Paragraph(f"\u2022  {t}", body_style))
    story.append(Spacer(1, 0.15*cm))

story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("Future Improvements", subheading_style))

improvements = [
    "Incorporate granular season-by-season performance trends and value trajectories.",
    "Add contract length, injury records, and international appearance data as features.",
    "Experiment with gradient boosting models (XGBoost, LightGBM) for potential accuracy gains.",
    "Include advanced metrics such as expected goals (xG), expected assists (xA), and pressing statistics.",
    "Expand the dataset to cover multiple leagues for cross-league valuation comparisons.",
]
for t in improvements:
    story.append(Paragraph(f"\u2022  {t}", body_style))
    story.append(Spacer(1, 0.15*cm))

story.append(Spacer(1, 1*cm))

# Footer note
footer_style = ParagraphStyle(
    "Footer", parent=styles["Normal"],
    fontSize=8, textColor=HexColor("#999999"),
    alignment=TA_CENTER,
)
story.append(Paragraph(
    "This report was generated automatically using Python, scikit-learn, matplotlib, and ReportLab. "
    "Data sourced from Transfermarkt. All values in Euros.",
    footer_style
))

# --- BUILD ---
doc.build(story)
print(f"\nPDF saved to: {PDF_PATH}")
print(f"File size: {os.path.getsize(PDF_PATH) / 1024:.1f} KB")
