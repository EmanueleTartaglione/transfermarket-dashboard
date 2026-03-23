#!/usr/bin/env python3
"""
Feature engineering script for Premier League market value prediction.

Reads all scraped CSVs and produces a rich feature set saved to model_features.csv.
Run: python3 build_features.py
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TODAY = datetime(2026, 3, 22)

TABLE_TO_PLAYER_CLUB = {
    "Arsenal": "Arsenal FC",
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Aston Villa": "Aston Villa",
    "Liverpool": "Liverpool FC",
    "Brighton": "Brighton & Hove Albion",
    "Newcastle": "Newcastle United",
    "Chelsea": "Chelsea FC",
    "Tottenham": "Tottenham Hotspur",
    "Fulham": "Fulham FC",
    "Brentford": "Brentford FC",
    "Bournemouth": "AFC Bournemouth",
    "Crystal Palace": "Crystal Palace",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Everton": "Everton FC",
    "Nott'm Forest": "Nottingham Forest",
    "Leicester": "Leicester City",
    "Ipswich": "Ipswich Town",
    "Southampton": "Southampton FC",
    # Additional clubs present in current season data
    "Sunderland": "Sunderland AFC",
    "Leeds": "Leeds United",
    "Burnley": "Burnley FC",
}


def read_csv(filename: str):
    """Read a CSV from the project directory; return None if missing."""
    path = os.path.join(PROJECT_DIR, filename)
    if not os.path.exists(path):
        print(f"  [SKIP] {filename} not found")
        return None
    df = pd.read_csv(path)
    print(f"  [OK]   {filename}: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


# =========================================================================
# 1. Load data
# =========================================================================
print("=" * 60)
print("Loading data...")
print("=" * 60)

players = read_csv("premier_league_players.csv")
stats = read_csv("player_stats.csv")
seasons = read_csv("player_seasons.csv")
value_history = read_csv("player_value_history.csv")
league_table = read_csv("premier_league_table.csv")
positions = read_csv("player_positions.csv")
extra_stats = read_csv("player_extra_stats.csv")
salaries = read_csv("player_salaries.csv")

if players is None:
    raise FileNotFoundError("premier_league_players.csv is required")

# =========================================================================
# 2. Base dataframe — one row per player
# =========================================================================
print("\n" + "=" * 60)
print("Building features...")
print("=" * 60)

df = players[["name", "club", "age", "date_of_birth", "height_m", "foot",
              "market_value_eur", "contract_expires"]].copy()

# ---------------------------------------------------------------------------
# 2a. Age features
# ---------------------------------------------------------------------------
print("  -> Age features")
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["age_squared"] = df["age"] ** 2
df["years_to_peak"] = df["age"] - 27
df["is_young_talent"] = (df["age"] <= 23).astype(int)
df["is_veteran"] = (df["age"] >= 31).astype(int)

# ---------------------------------------------------------------------------
# 2b. Contract features
# ---------------------------------------------------------------------------
print("  -> Contract features")


def parse_contract_years(val: str):
    """Parse DD/MM/YYYY contract expiry and return years remaining from TODAY."""
    if pd.isna(val) or str(val).strip() in ("-", "", "nan"):
        return np.nan
    try:
        expiry = datetime.strptime(str(val).strip(), "%d/%m/%Y")
        delta = (expiry - TODAY).days / 365.25
        return max(delta, 0.0)
    except ValueError:
        return np.nan


df["contract_years_remaining"] = df["contract_expires"].apply(parse_contract_years)
df["contract_expiring_soon"] = (df["contract_years_remaining"] <= 1).astype(int)
df["contract_long"] = (df["contract_years_remaining"] >= 4).astype(int)

# Where contract is missing, flags should be NaN (not 0)
mask_no_contract = df["contract_years_remaining"].isna()
df.loc[mask_no_contract, "contract_expiring_soon"] = np.nan
df.loc[mask_no_contract, "contract_long"] = np.nan

# ---------------------------------------------------------------------------
# 2c. Physical & profile features
# ---------------------------------------------------------------------------
print("  -> Physical features")
df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")
df["is_right_footed"] = (df["foot"].str.lower() == "right").astype(int)

# ---------------------------------------------------------------------------
# 3. Performance rate stats (from player_stats.csv)
# ---------------------------------------------------------------------------
if stats is not None:
    print("  -> Career performance rate stats")
    stat_cols = ["name", "total_appearances", "total_goals", "total_assists",
                 "total_yellow_cards", "total_minutes"]
    s = stats[stat_cols].copy()
    for c in stat_cols[1:]:
        s[c] = pd.to_numeric(s[c], errors="coerce")

    nineties = s["total_minutes"] / 90.0
    nineties_safe = nineties.replace(0, np.nan)

    s["goals_per_90"] = s["total_goals"] / nineties_safe
    s["assists_per_90"] = s["total_assists"] / nineties_safe
    s["goal_contributions_per_90"] = (s["total_goals"] + s["total_assists"]) / nineties_safe
    s["yellows_per_90"] = s["total_yellow_cards"] / nineties_safe

    apps_safe = s["total_appearances"].replace(0, np.nan)
    s["minutes_per_appearance"] = s["total_minutes"] / apps_safe

    merge_cols = ["name", "total_appearances", "total_goals", "total_assists",
                  "total_minutes", "goals_per_90", "assists_per_90",
                  "goal_contributions_per_90", "yellows_per_90",
                  "minutes_per_appearance"]
    df = df.merge(s[merge_cols], on="name", how="left")

# ---------------------------------------------------------------------------
# 4. Current season stats (from player_seasons.csv, 25/26 Premier League)
# ---------------------------------------------------------------------------
if seasons is not None:
    print("  -> Current season (25/26 PL) stats")
    pl = seasons[
        (seasons["season"] == "25/26") & (seasons["competition"] == "Premier League")
    ].copy()

    for c in ["appearances", "goals", "assists", "minutes_played", "yellow_cards", "red_cards"]:
        pl[c] = pd.to_numeric(pl[c], errors="coerce")

    pl = pl.rename(columns={
        "appearances": "season_apps",
        "goals": "season_goals",
        "assists": "season_assists",
        "minutes_played": "season_minutes",
        "yellow_cards": "season_yellows",
        "red_cards": "season_reds",
    })

    nineties_s = pl["season_minutes"] / 90.0
    nineties_s_safe = nineties_s.replace(0, np.nan)
    pl["season_goals_per_90"] = pl["season_goals"] / nineties_s_safe
    pl["season_assists_per_90"] = pl["season_assists"] / nineties_s_safe

    # Max possible minutes per club: matches * 90
    if league_table is not None:
        lt = league_table[["club", "matches"]].copy()
        lt["club"] = lt["club"].map(TABLE_TO_PLAYER_CLUB).fillna(lt["club"])
        lt["max_possible_minutes"] = pd.to_numeric(lt["matches"], errors="coerce") * 90
        pl = pl.merge(lt[["club", "max_possible_minutes"]], on="club", how="left")
        pl["season_minutes_ratio"] = pl["season_minutes"] / pl["max_possible_minutes"].replace(0, np.nan)
    else:
        pl["season_minutes_ratio"] = np.nan

    pl["is_regular_starter"] = (pl["season_minutes_ratio"] > 0.6).astype(int)
    pl.loc[pl["season_minutes_ratio"].isna(), "is_regular_starter"] = np.nan

    season_merge = ["name", "season_apps", "season_goals", "season_assists",
                    "season_minutes", "season_yellows", "season_reds",
                    "season_goals_per_90", "season_assists_per_90",
                    "season_minutes_ratio", "is_regular_starter"]
    df = df.merge(pl[season_merge], on="name", how="left")

# ---------------------------------------------------------------------------
# 5. European competition features (from player_seasons.csv — full career)
# ---------------------------------------------------------------------------
if seasons is not None:
    print("  -> European competition features")

    def _has_comp(group, keyword):
        return int(group["competition"].str.contains(keyword, case=False, na=False).any())

    euro = seasons.groupby("name").apply(
        lambda g: pd.Series({
            "has_champions_league": _has_comp(g, "Champions League"),
            "has_europa_league": int(
                g["competition"].str.contains("Europa League", case=False, na=False).any()
                & ~g["competition"].str.contains("Qualif", case=False, na=False).all()
            ),
            "has_conference_league": _has_comp(g, "Conference"),
        })
    ).reset_index()

    # European tier: best ever (1=CL, 2=EL, 3=ECL, 4=none)
    def _euro_tier(row):
        if row["has_champions_league"]:
            return 1
        elif row["has_europa_league"]:
            return 2
        elif row["has_conference_league"]:
            return 3
        return 4
    euro["european_tier"] = euro.apply(_euro_tier, axis=1)

    # CL career appearances & goals
    cl_rows = seasons[seasons["competition"].str.contains("Champions League", case=False, na=False)].copy()
    for c in ["appearances", "goals"]:
        cl_rows[c] = pd.to_numeric(cl_rows[c], errors="coerce")
    cl_agg = cl_rows.groupby("name").agg(
        cl_appearances=("appearances", "sum"),
        cl_goals=("goals", "sum"),
    ).reset_index()

    euro = euro.merge(cl_agg, on="name", how="left")
    euro["cl_appearances"] = euro["cl_appearances"].fillna(0).astype(int)
    euro["cl_goals"] = euro["cl_goals"].fillna(0).astype(int)

    df = df.merge(euro, on="name", how="left")

# ---------------------------------------------------------------------------
# 6. Club prestige features (from league table)
# ---------------------------------------------------------------------------
if league_table is not None:
    print("  -> Club prestige features")
    lt = league_table.copy()
    lt["club"] = lt["club"].map(TABLE_TO_PLAYER_CLUB).fillna(lt["club"])
    lt = lt.rename(columns={"position": "club_league_position", "points": "club_points"})
    for c in ["club_league_position", "club_points"]:
        lt[c] = pd.to_numeric(lt[c], errors="coerce")

    lt["is_top6_club"] = (lt["club_league_position"] <= 6).astype(int)
    lt["is_relegation_club"] = (lt["club_league_position"] >= 18).astype(int)

    prestige_cols = ["club", "club_league_position", "club_points",
                     "is_top6_club", "is_relegation_club"]
    df = df.merge(lt[prestige_cols], on="club", how="left")

    # Club squad value aggregates
    squad_val = df.groupby("club")["market_value_eur"].agg(
        club_squad_value="sum",
        club_avg_value="mean",
    ).reset_index()
    df = df.merge(squad_val, on="club", how="left")

# ---------------------------------------------------------------------------
# 7. Value trajectory features (from player_value_history.csv)
# ---------------------------------------------------------------------------
if value_history is not None:
    print("  -> Value trajectory features")
    vh = value_history.copy()
    vh["market_value_eur"] = pd.to_numeric(vh["market_value_eur"], errors="coerce")
    vh["date"] = pd.to_datetime(vh["date"], format="%d/%m/%Y", errors="coerce")

    # Latest value per player (should match players table, but just in case)
    vh_sorted = vh.sort_values("date")

    def compute_value_features(group):
        group = group.sort_values("date")
        latest = group.iloc[-1]
        current_val = latest["market_value_eur"]

        result = {}

        # Value 12 months ago
        target_date = TODAY - pd.DateOffset(months=12)
        past = group[group["date"] <= target_date]
        if len(past) > 0:
            result["value_12m_ago"] = past.iloc[-1]["market_value_eur"]
            result["value_change_12m"] = current_val - result["value_12m_ago"]
            old_safe = result["value_12m_ago"] if result["value_12m_ago"] != 0 else np.nan
            result["value_change_pct_12m"] = result["value_change_12m"] / old_safe * 100
        else:
            result["value_12m_ago"] = np.nan
            result["value_change_12m"] = np.nan
            result["value_change_pct_12m"] = np.nan

        # Peak value
        result["value_peak"] = group["market_value_eur"].max()
        peak_safe = result["value_peak"] if result["value_peak"] != 0 else np.nan
        result["value_at_peak_ratio"] = current_val / peak_safe

        # Value trend: slope of linear regression over last 3 years
        three_years_ago = TODAY - pd.DateOffset(years=3)
        recent = group[group["date"] >= three_years_ago].copy()
        if len(recent) >= 2:
            x = (recent["date"] - recent["date"].iloc[0]).dt.days.values.astype(float)
            y = recent["market_value_eur"].values.astype(float)
            if x[-1] > 0:
                # Simple linear regression slope (EUR per day)
                coeffs = np.polyfit(x, y, 1)
                result["value_trend"] = coeffs[0]  # EUR per day
            else:
                result["value_trend"] = np.nan
        else:
            result["value_trend"] = np.nan

        return pd.Series(result)

    val_feats = vh.groupby("name").apply(compute_value_features).reset_index()
    df = df.merge(val_feats, on="name", how="left")

# ---------------------------------------------------------------------------
# 8. Position features (from player_positions.csv)
# ---------------------------------------------------------------------------
if positions is not None:
    print("  -> Position features")
    pos = positions[["name", "main_position", "all_positions"]].copy()

    # Position group mapping
    POSITION_GROUP_MAP = {
        "Goalkeeper": "Goalkeeper",
        "Centre-Back": "Defender",
        "Left-Back": "Defender",
        "Right-Back": "Defender",
        "Defensive Midfield": "Midfielder",
        "Central Midfield": "Midfielder",
        "Attacking Midfield": "Midfielder",
        "Left Midfield": "Midfielder",
        "Right Midfield": "Midfielder",
        "Left Winger": "Forward",
        "Right Winger": "Forward",
        "Centre-Forward": "Forward",
        "Second Striker": "Forward",
    }

    pos["position_group"] = pos["main_position"].map(POSITION_GROUP_MAP).fillna("Unknown")

    # Number of positions and versatility
    pos["num_positions"] = pos["all_positions"].apply(
        lambda x: len(str(x).split(",")) if pd.notna(x) and str(x).strip() else 0
    )
    pos["is_versatile"] = (pos["num_positions"] >= 3).astype(int)

    # One-hot for position group
    pos_dummies = pd.get_dummies(pos["position_group"], prefix="pos")
    pos = pd.concat([pos, pos_dummies], axis=1)

    # Columns to merge
    pos_merge_cols = ["name", "main_position", "position_group",
                      "num_positions", "is_versatile"]
    pos_merge_cols += [c for c in pos_dummies.columns]
    df = df.merge(pos[pos_merge_cols], on="name", how="left")

# ---------------------------------------------------------------------------
# 9. Extra stats / transfer history (from player_extra_stats.csv — partial)
# ---------------------------------------------------------------------------
if extra_stats is not None:
    print("  -> Extra stats (transfer history, international, injuries)")
    ex = extra_stats[["name", "international_caps", "international_goals",
                       "career_injuries", "career_days_injured",
                       "num_transfers", "total_transfer_fees_eur",
                       "highest_fee_eur", "num_clubs",
                       "page_views", "years_at_current_club"]].copy()
    for c in ex.columns:
        if c != "name":
            ex[c] = pd.to_numeric(ex[c], errors="coerce")

    df = df.merge(ex, on="name", how="left")

# ---------------------------------------------------------------------------
# 10. Salary features (from player_salaries.csv — may not exist)
# ---------------------------------------------------------------------------
if salaries is not None:
    print("  -> Salary features")
    sal = salaries.copy()
    if "weekly_wage_eur" in sal.columns and "name" in sal.columns:
        sal["weekly_wage_eur"] = pd.to_numeric(sal["weekly_wage_eur"], errors="coerce")
        # SoFIFA uses full legal names; Transfermarkt uses common names.
        # Match by: 1) exact name, 2) last-name + club fuzzy match
        import unicodedata
        def normalize(s):
            s = str(s).strip().lower()
            s = unicodedata.normalize("NFKD", s)
            s = "".join(c for c in s if not unicodedata.combining(c))
            return s

        # SoFIFA club -> Transfermarkt club mapping
        SOFIFA_CLUB_MAP = {
            "arsenal": "arsenal fc", "manchester city": "manchester city",
            "manchester united": "manchester united", "liverpool": "liverpool fc",
            "chelsea": "chelsea fc", "tottenham hotspur": "tottenham hotspur",
            "aston villa": "aston villa", "newcastle united": "newcastle united",
            "brighton & hove albion": "brighton & hove albion",
            "west ham united": "west ham united", "crystal palace": "crystal palace",
            "fulham fc": "fulham fc", "everton": "everton fc",
            "brentford": "brentford fc", "afc bournemouth": "afc bournemouth",
            "nottingham forest": "nottingham forest",
            "wolverhampton wanderers": "wolverhampton wanderers",
            "burnley": "burnley fc", "leeds united": "leeds united",
            "sunderland": "sunderland afc",
        }

        # Build lookup structures
        sal_by_name = {}  # normalised full name -> wage
        sal_by_club_surname = {}  # (norm_club, norm_surname) -> wage
        sal_rows = []  # for substring matching
        for _, row in sal.iterrows():
            nname = normalize(row["name"])
            wage = row["weekly_wage_eur"]
            sal_by_name[nname] = wage
            nclub = normalize(str(row.get("club", "")))
            mapped_club = SOFIFA_CLUB_MAP.get(nclub, nclub)
            parts = str(row["name"]).strip().split()
            if len(parts) > 1:
                surname = normalize(parts[-1])
                sal_by_club_surname[(mapped_club, surname)] = wage
            sal_rows.append((nname, mapped_club, wage))

        def find_wage(player_name, player_club):
            nname = normalize(player_name)
            nclub = normalize(str(player_club))
            # 1. Exact normalised name match
            if nname in sal_by_name:
                return sal_by_name[nname]
            # 2. Surname + club match (requires club agreement)
            parts = str(player_name).strip().split()
            for part in reversed(parts):
                npart = normalize(part)
                if len(npart) >= 3:  # avoid matching on short names
                    key = (nclub, npart)
                    if key in sal_by_club_surname:
                        return sal_by_club_surname[key]
            # 3. Substring match: player_name contained in sal_name (or vice versa), same club
            for sal_name, sal_club, wage in sal_rows:
                if sal_club != nclub:
                    continue
                if nname in sal_name or sal_name in nname:
                    return wage
                # Check if any substantial part of player name is in sal name
                for part in parts:
                    npart = normalize(part)
                    if len(npart) >= 4 and npart in sal_name:
                        return wage
            return np.nan

        df["weekly_wage_eur"] = df.apply(
            lambda r: find_wage(r["name"], r["club"]), axis=1
        )
        df["wage_to_value_ratio"] = df["weekly_wage_eur"] * 52 / df["market_value_eur"]
        matched = df["weekly_wage_eur"].notna().sum()
        print(f"    Matched wages for {matched}/{len(df)} players ({matched/len(df)*100:.0f}%)")
    else:
        print("    [WARN] player_salaries.csv found but missing expected columns")
else:
    print("  -> Salary features: SKIPPED (player_salaries.csv not found)")

# =========================================================================
# Clean up helper columns
# =========================================================================
print("\n" + "=" * 60)
print("Finalising...")
print("=" * 60)

# Drop raw text columns not useful for modelling
drop_cols = ["date_of_birth", "contract_expires", "foot"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Ensure market_value_eur is numeric
df["market_value_eur"] = pd.to_numeric(df["market_value_eur"], errors="coerce")

# =========================================================================
# Save
# =========================================================================
out_path = os.path.join(PROJECT_DIR, "model_features.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved {df.shape[0]} rows x {df.shape[1]} columns -> {out_path}")

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 60)
print("Feature summary")
print("=" * 60)

# Group features by category for display
feature_groups = {
    "Target": ["market_value_eur"],
    "Identity": ["name", "club"],
    "Age": ["age", "age_squared", "years_to_peak", "is_young_talent", "is_veteran"],
    "Contract": ["contract_years_remaining", "contract_expiring_soon", "contract_long"],
    "Physical": ["height_m", "is_right_footed"],
    "Career performance": ["total_appearances", "total_goals", "total_assists",
                           "total_minutes", "goals_per_90", "assists_per_90",
                           "goal_contributions_per_90", "yellows_per_90",
                           "minutes_per_appearance"],
    "Current season": ["season_apps", "season_goals", "season_assists",
                       "season_minutes", "season_yellows", "season_reds",
                       "season_goals_per_90", "season_assists_per_90",
                       "season_minutes_ratio", "is_regular_starter"],
    "European": ["has_champions_league", "has_europa_league",
                 "has_conference_league", "european_tier",
                 "cl_appearances", "cl_goals"],
    "Club prestige": ["club_league_position", "club_points",
                      "club_squad_value", "club_avg_value",
                      "is_top6_club", "is_relegation_club"],
    "Value trajectory": ["value_12m_ago", "value_change_12m", "value_change_pct_12m",
                         "value_peak", "value_at_peak_ratio", "value_trend"],
    "Position": ["main_position", "position_group", "num_positions", "is_versatile"],
    "Extra / transfer": ["international_caps", "international_goals",
                         "career_injuries", "career_days_injured",
                         "num_transfers", "total_transfer_fees_eur",
                         "highest_fee_eur", "num_clubs",
                         "page_views", "years_at_current_club"],
    "Salary": ["weekly_wage_eur", "wage_to_value_ratio"],
}

total_features = 0
for group_name, cols in feature_groups.items():
    present = [c for c in cols if c in df.columns]
    if not present:
        continue
    print(f"\n  {group_name} ({len(present)} features):")
    for c in present:
        pct_missing = df[c].isna().mean() * 100
        print(f"    {c:<35s}  missing: {pct_missing:5.1f}%")
    total_features += len(present)

# Position one-hot columns
pos_oh = [c for c in df.columns if c.startswith("pos_")]
if pos_oh:
    print(f"\n  Position one-hot ({len(pos_oh)} features):")
    for c in pos_oh:
        pct_missing = df[c].isna().mean() * 100
        print(f"    {c:<35s}  missing: {pct_missing:5.1f}%")
    total_features += len(pos_oh)

print(f"\nTotal features: {total_features}")
print(f"Total rows:     {df.shape[0]}")
print("Done.")
