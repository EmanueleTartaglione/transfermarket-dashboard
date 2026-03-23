"""
Premier League Player Market Value Analysis Dashboard
=====================================================
A comprehensive Streamlit dashboard for exploring Transfermarkt data
on Premier League players, clubs, value trends, and predictive insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Premier League Value Analysis",
    page_icon="\u26BD",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent

# Color palette
COLORS = {
    "primary": "#1B1464",
    "secondary": "#3D5A80",
    "accent": "#EE6C4D",
    "bg": "#F7F9FC",
    "text": "#293241",
    "positive": "#06D6A0",
    "negative": "#EF476F",
}

# Canonical position order
POSITION_ORDER = [
    "Goalkeeper",
    "Right-Back",
    "Centre-Back",
    "Left-Back",
    "Defensive Midfield",
    "Right Midfield",
    "Central Midfield",
    "Left Midfield",
    "Attacking Midfield",
    "Right Winger",
    "Left Winger",
    "Centre-Forward",
]

POSITION_COLORS = {
    "Goalkeeper": "#FDCB58",
    "Centre-Back": "#3D5A80",
    "Left-Back": "#5B9BD5",
    "Right-Back": "#70B0E0",
    "Defensive Midfield": "#457B9D",
    "Central Midfield": "#2A9D8F",
    "Attacking Midfield": "#E9C46A",
    "Left Midfield": "#76C893",
    "Right Midfield": "#52B788",
    "Left Winger": "#F4845F",
    "Right Winger": "#F4A261",
    "Centre-Forward": "#E76F51",
}

POSITION_GROUPS = {
    "Goalkeeper": "GK",
    "Centre-Back": "DEF",
    "Left-Back": "DEF",
    "Right-Back": "DEF",
    "Defensive Midfield": "MID",
    "Central Midfield": "MID",
    "Attacking Midfield": "MID",
    "Left Midfield": "MID",
    "Right Midfield": "MID",
    "Left Winger": "FWD",
    "Right Winger": "FWD",
    "Centre-Forward": "FWD",
}

POSITION_GROUP_ORDER = ["GK", "DEF", "MID", "FWD"]

# Position label with sort prefix: "01 · Goalkeeper", "02 · Right-Back", etc.
POS_LABEL_MAP = {p: f"{i+1:02d} · {p}" for i, p in enumerate(POSITION_ORDER)}

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B1464 0%, #3D5A80 100%);
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4,
    section[data-testid="stSidebar"] .stMarkdown h5,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] [class*="caption"],
    section[data-testid="stSidebar"] [class*="Caption"] {
        color: white !important;
    }
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #EE6C4D;
    }
    div[data-testid="stMetric"] label {
        color: #3D5A80 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1B1464 !important;
        font-weight: 700 !important;
    }
    /* Dataframes */
    .stDataFrame {border-radius: 10px; overflow: hidden;}
    /* Tabs */
    .stTabs [data-baseweb="tab"] {font-weight: 600;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data Loading & Parsing
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading player data...")
def load_players() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "premier_league_players.csv")
    # market_value_eur is already numeric in the CSV
    df["value"] = pd.to_numeric(df["market_value_eur"], errors="coerce").fillna(0)
    df["height"] = pd.to_numeric(df["height_m"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["joined_date"] = pd.to_datetime(df["joined"], dayfirst=True, errors="coerce")
    df["contract_until"] = pd.to_datetime(
        df["contract_expires"].replace("-", pd.NaT), dayfirst=True, errors="coerce"
    )
    df["position_group"] = df["position"].map(POSITION_GROUPS).fillna("Other")
    # Add position sort key
    pos_order_map = {p: i for i, p in enumerate(POSITION_ORDER)}
    df["position_sort"] = df["position"].map(pos_order_map).fillna(99)
    return df


@st.cache_data(show_spinner="Loading stats...")
def load_stats() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "player_stats.csv", low_memory=False)
    for c in [
        "total_appearances",
        "total_goals",
        "total_assists",
        "total_yellow_cards",
        "total_red_cards",
        "total_minutes",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data(show_spinner="Loading season data...")
def load_seasons() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "player_seasons.csv")
    for c in ["appearances", "goals", "assists", "yellow_cards", "red_cards", "minutes_played"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data(show_spinner="Loading value history...")
def load_value_history() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "player_value_history.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["value"] = pd.to_numeric(df["market_value_eur"], errors="coerce").fillna(0)
    return df.sort_values(["player_id", "date"])


@st.cache_data(show_spinner="Loading player positions...")
def load_positions() -> pd.DataFrame:
    path = DATA_DIR / "player_positions.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df[["name", "all_positions"]].drop_duplicates("name")
    return pd.DataFrame(columns=["name", "all_positions"])


@st.cache_data(show_spinner="Loading league table...")
def load_league_table() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "premier_league_table.csv")
    for c in ["position", "matches", "wins", "draws", "losses", "goals_for",
              "goals_against", "goal_difference", "points"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df.sort_values("position")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_value(v: float) -> str:
    """Format a euro value for display -- always in millions."""
    if pd.isna(v) or v == 0:
        return "-"
    sign = "-" if v < 0 else ""
    av = abs(v)
    millions = av / 1_000_000
    if millions >= 1:
        return f"{sign}\u20AC{millions:.1f}M"
    return f"{sign}\u20AC{millions:.2f}M"


def fmt_value_short(v: float) -> str:
    if pd.isna(v) or v == 0:
        return "-"
    sign = "-" if v < 0 else ""
    av = abs(v)
    if av >= 1_000_000_000:
        return f"{sign}\u20AC{av / 1_000_000_000:.2f}B"
    millions = av / 1_000_000
    if millions >= 1:
        return f"{sign}\u20AC{millions:.0f}M"
    return f"{sign}\u20AC{millions:.2f}M"


def player_display_name(name: str, club: str) -> str:
    """Format player name as 'Name (Club)' for charts and dropdowns."""
    return f"{name} ({club})"


def plotly_defaults(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        margin=dict(l=40, r=40, t=50, b=40),
        hoverlabel=dict(bgcolor="white", font_size=12),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def get_current_season_stats(seasons_df: pd.DataFrame, player_name: str) -> dict:
    """Get current season (25/26) Premier League stats for a player."""
    mask = (
        (seasons_df["name"] == player_name)
        & (seasons_df["season"] == "25/26")
        & (seasons_df["competition"] == "Premier League")
    )
    row = seasons_df[mask]
    if row.empty:
        return {"season_apps": 0, "season_goals": 0, "season_assists": 0}
    row = row.iloc[0]
    return {
        "season_apps": int(row.get("appearances", 0)),
        "season_goals": int(row.get("goals", 0)),
        "season_assists": int(row.get("assists", 0)),
    }


def position_sort_key(position_series: pd.Series) -> pd.Series:
    """Return a sort key series for positions based on POSITION_ORDER."""
    pos_map = {p: i for i, p in enumerate(POSITION_ORDER)}
    return position_series.map(pos_map).fillna(99)


# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------
players = load_players()
stats = load_stats()
seasons = load_seasons()
value_history = load_value_history()
league_table = load_league_table()
positions_data = load_positions()

# Merge stats into players for convenience
players_full = players.merge(
    stats[["name", "player_id", "total_appearances", "total_goals", "total_assists",
           "total_yellow_cards", "total_red_cards", "total_minutes"]],
    on="name",
    how="left",
)

for c in ["total_appearances", "total_goals", "total_assists", "total_minutes",
          "total_yellow_cards", "total_red_cards"]:
    if c in players_full.columns:
        players_full[c] = players_full[c].fillna(0).astype(int)

# Build current season stats lookup (25/26 Premier League only)
current_season_pl = seasons[
    (seasons["season"] == "25/26") & (seasons["competition"] == "Premier League")
].copy()
current_season_agg = current_season_pl.groupby("name").agg(
    season_apps=("appearances", "sum"),
    season_goals=("goals", "sum"),
    season_assists=("assists", "sum"),
).reset_index()

players_full = players_full.merge(current_season_agg, on="name", how="left")
for c in ["season_apps", "season_goals", "season_assists"]:
    players_full[c] = players_full[c].fillna(0).astype(int)

# Merge multi-position data
if not positions_data.empty:
    players_full = players_full.merge(positions_data, on="name", how="left")
    players_full["all_positions"] = players_full["all_positions"].fillna(players_full["position"])
else:
    players_full["all_positions"] = players_full["position"]

# Add player status flag (Active / Fringe / Inactive-Injured / Loaned Out)
# Detect loaned-out: players whose 25/26 season entries are at a different club
_cur_season = seasons[seasons["season"] == "25/26"].copy()
_non_own_club = _cur_season.merge(
    players_full[["name", "club"]].drop_duplicates("name"), on="name", how="inner",
    suffixes=("_s", "_r")
)
_loaned_names = set(
    _non_own_club[
        (_non_own_club["club_s"] != _non_own_club["club_r"])
        & (~_non_own_club["club_s"].str.contains("U21|U18|U23", na=False))
    ]["name"].unique()
)

def _classify_status(row):
    if row["name"] in _loaned_names and row.get("season_apps", 0) == 0:
        return "Loaned Out"
    if row.get("season_apps", 0) == 0:
        return "Inactive/Injured"
    if row.get("season_apps", 0) <= 3:
        return "Fringe"
    return "Active"

players_full["player_status"] = players_full.apply(_classify_status, axis=1)

# Add display name column
players_full["display_name"] = players_full.apply(
    lambda r: player_display_name(r["name"], r["club"]), axis=1
)

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## \u26BD Premier League")
    st.markdown("##### Value Analysis Dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "Overview Dashboard",
            "League Table",
            "Player Explorer",
            "Club Analysis",
            "Value Trends",
            "Head-to-Head",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Data sourced from Transfermarkt")
    st.caption(f"{len(players)} players \u00B7 {players['club'].nunique()} clubs")


# ===================================================================
# PAGE -- LEAGUE TABLE
# ===================================================================
if page == "League Table":
    st.markdown("# 2025/26 Premier League Standings")
    st.markdown("Current league table and squad values by league position.")

    lt = league_table.copy()

    # Color-code rows based on position
    def row_color(pos):
        if pos <= 4:
            return "background-color: rgba(6, 214, 160, 0.15)"  # Champions League - green
        elif pos == 5:
            return "background-color: rgba(69, 123, 157, 0.15)"  # Europa League - blue
        elif pos == 6:
            return "background-color: rgba(233, 196, 106, 0.15)"  # Conference League - yellow
        elif pos >= 18:
            return "background-color: rgba(239, 71, 111, 0.15)"  # Relegation - red
        return ""

    display_lt = lt[["position", "club", "matches", "wins", "draws", "losses",
                      "goals_for", "goals_against", "goal_difference", "points"]].copy()
    display_lt.columns = ["Pos", "Club", "MP", "W", "D", "L", "GF", "GA", "GD", "Pts"]
    display_lt.index = range(1, len(display_lt) + 1)

    def highlight_rows(row):
        pos = row["Pos"]
        if pos <= 4:
            return ["background-color: rgba(6, 214, 160, 0.35)"] * len(row)
        elif pos == 5:
            return ["background-color: rgba(69, 123, 157, 0.35)"] * len(row)
        elif pos == 6:
            return ["background-color: rgba(233, 196, 106, 0.35)"] * len(row)
        elif pos >= 18:
            return ["background-color: rgba(239, 71, 111, 0.35)"] * len(row)
        return [""] * len(row)

    styled_table = display_lt.style.apply(highlight_rows, axis=1)
    st.dataframe(styled_table, use_container_width=True, height=750, hide_index=True)

    # Legend
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.markdown('<span style="display:inline-block;width:16px;height:16px;background:#06D6A0;border-radius:3px;vertical-align:middle;margin-right:6px;"></span> **Pos 1-4:** Champions League', unsafe_allow_html=True)
    lc2.markdown('<span style="display:inline-block;width:16px;height:16px;background:#457B9D;border-radius:3px;vertical-align:middle;margin-right:6px;"></span> **Pos 5:** Europa League', unsafe_allow_html=True)
    lc3.markdown('<span style="display:inline-block;width:16px;height:16px;background:#E9C46A;border-radius:3px;vertical-align:middle;margin-right:6px;"></span> **Pos 6:** Conference League', unsafe_allow_html=True)
    lc4.markdown('<span style="display:inline-block;width:16px;height:16px;background:#EF476F;border-radius:3px;vertical-align:middle;margin-right:6px;"></span> **Pos 18-20:** Relegation', unsafe_allow_html=True)

    # Bar chart of squad value by league position
    st.markdown("---")
    st.markdown("### Total Squad Market Value by League Position")

    # Map short table names to full player CSV names
    TABLE_TO_PLAYER_CLUB = {
        "Arsenal": "Arsenal FC", "Man City": "Manchester City", "Man Utd": "Manchester United",
        "Aston Villa": "Aston Villa", "Liverpool": "Liverpool FC", "Chelsea": "Chelsea FC",
        "Brentford": "Brentford FC", "Everton": "Everton FC", "Fulham": "Fulham FC",
        "Brighton": "Brighton & Hove Albion", "Newcastle": "Newcastle United",
        "Bournemouth": "AFC Bournemouth", "Sunderland": "Sunderland AFC",
        "Crystal Palace": "Crystal Palace", "Leeds": "Leeds United",
        "Tottenham": "Tottenham Hotspur", "Nott'm Forest": "Nottingham Forest",
        "West Ham": "West Ham United", "Burnley": "Burnley FC", "Wolves": "Wolverhampton Wanderers",
    }
    # Calculate squad values and order by league position
    club_values = players.groupby("club")["value"].sum().reset_index()
    club_values.columns = ["club_full", "squad_value"]

    lt_mapped = lt[["position", "club"]].copy()
    lt_mapped["club_full"] = lt_mapped["club"].map(TABLE_TO_PLAYER_CLUB)

    # Merge with league table to get position ordering
    lt_vals = lt_mapped.merge(club_values, on="club_full", how="left")
    lt_vals["squad_value"] = lt_vals["squad_value"].fillna(0)
    lt_vals = lt_vals.sort_values("position")
    lt_vals["value_display"] = lt_vals["squad_value"].apply(fmt_value_short)

    fig_sv = px.bar(
        lt_vals,
        x="club",
        y="squad_value",
        text="value_display",
        color_discrete_sequence=[COLORS["accent"]],
        labels={"squad_value": "Total Squad Value (\u20AC)", "club": "Club"},
    )
    fig_sv.update_traces(textposition="outside", textfont_size=10)
    fig_sv.update_layout(xaxis={"categoryorder": "array", "categoryarray": lt_vals["club"].tolist()})
    plotly_defaults(fig_sv)
    fig_sv.update_layout(height=550, xaxis_tickangle=-45)
    st.plotly_chart(fig_sv, use_container_width=True)


# ===================================================================
# PAGE -- OVERVIEW DASHBOARD
# ===================================================================
elif page == "Overview Dashboard":
    st.markdown("# Overview Dashboard")
    st.markdown("A bird's-eye view of the Premier League player market.")

    # --- KPI row ---
    k1, k2, k3, k4 = st.columns(4)
    total_players = len(players)
    avg_value = players["value"].mean()
    total_value = players["value"].sum()
    avg_age = players["age"].mean()
    k1.metric("Total Players", f"{total_players:,}")
    k2.metric("Avg Market Value", fmt_value(avg_value))
    k3.metric("Total Squad Value", fmt_value_short(total_value))
    k4.metric("Avg Age", f"{avg_age:.1f}")

    st.markdown("")

    # --- Club ranking ---
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### Club Ranking by Total Squad Value")
        club_vals = (
            players.groupby("club")
            .agg(total_value=("value", "sum"), num_players=("name", "count"), avg_value=("value", "mean"))
            .sort_values("total_value", ascending=False)
            .reset_index()
        )
        club_vals["total_millions"] = club_vals["total_value"] / 1_000_000
        club_vals["avg_millions"] = club_vals["avg_value"] / 1_000_000
        club_vals.index = range(1, len(club_vals) + 1)
        st.dataframe(
            club_vals[["club", "num_players", "total_millions", "avg_millions"]].rename(
                columns={
                    "club": "Club",
                    "num_players": "Players",
                    "total_millions": "Total Value (€M)",
                    "avg_millions": "Avg Value (€M)",
                }
            ),
            use_container_width=True,
            height=500,
            column_config={
                "Total Value (€M)": st.column_config.NumberColumn(format="€%.0fM"),
                "Avg Value (€M)": st.column_config.NumberColumn(format="€%.1fM"),
            },
        )

    with col_right:
        st.markdown("### Position Breakdown")
        # Order by POSITION_ORDER
        pos_counts = players["position"].value_counts().reset_index()
        pos_counts.columns = ["Position", "Count"]
        pos_counts["sort_key"] = pos_counts["Position"].map(
            {p: i for i, p in enumerate(POSITION_ORDER)}
        ).fillna(99)
        pos_counts = pos_counts.sort_values("sort_key")
        fig_pie = px.pie(
            pos_counts,
            values="Count",
            names="Position",
            color="Position",
            color_discrete_map=POSITION_COLORS,
            hole=0.45,
            category_orders={"Position": POSITION_ORDER},
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label", textfont_size=10)
        plotly_defaults(fig_pie)
        fig_pie.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- Value distribution ---
    st.markdown("### Market Value Distribution")
    dist_tab1, dist_tab2 = st.tabs(["Histogram", "By Position Group"])

    with dist_tab1:
        fig_hist = px.histogram(
            players[players["value"] > 0],
            x="value",
            nbins=40,
            color_discrete_sequence=[COLORS["accent"]],
            labels={"value": "Market Value (\u20AC)"},
        )
        fig_hist.update_layout(yaxis_title="Number of Players", bargap=0.05)
        plotly_defaults(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

    with dist_tab2:
        fig_box = px.box(
            players[players["value"] > 0],
            x="position_group",
            y="value",
            color="position_group",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"value": "Market Value (\u20AC)", "position_group": "Position Group"},
            category_orders={"position_group": POSITION_GROUP_ORDER},
        )
        plotly_defaults(fig_box)
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)


# ===================================================================
# PAGE -- PLAYER EXPLORER
# ===================================================================
elif page == "Player Explorer":
    st.markdown("# Player Explorer")
    st.markdown("Search, filter, and drill into individual players.")

    # --- Filters ---
    with st.expander("Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sel_clubs = st.multiselect("Club", sorted(players["club"].unique()), default=[])
            sel_positions = st.multiselect(
                "Position",
                [p for p in POSITION_ORDER if p in players["position"].unique()],
                default=[],
            )
        with fc2:
            age_range = st.slider(
                "Age Range",
                int(players["age"].min()),
                int(players["age"].max()),
                (int(players["age"].min()), int(players["age"].max())),
            )
            max_val = int(players["value"].max())
            value_range = st.slider(
                "Value Range (\u20ACM)",
                0.0,
                max_val / 1_000_000,
                (0.0, max_val / 1_000_000),
                step=0.5,
            )
        with fc3:
            sel_nationality = st.multiselect(
                "Nationality", sorted(players["nationality"].unique()), default=[]
            )
            search_name = st.text_input("Search by Name")
            if "player_status" in players_full.columns:
                status_options = sorted(players_full["player_status"].unique())
                sel_status = st.multiselect("Player Status", status_options, default=[])

    # Apply filters
    mask = pd.Series(True, index=players_full.index)
    if sel_clubs:
        mask &= players_full["club"].isin(sel_clubs)
    if sel_positions:
        # Use all_positions for multi-position matching
        mask &= players_full["all_positions"].apply(
            lambda ap: any(p in str(ap) for p in sel_positions)
        )
    mask &= players_full["age"].between(*age_range)
    mask &= players_full["value"].between(value_range[0] * 1e6, value_range[1] * 1e6)
    if sel_nationality:
        mask &= players_full["nationality"].isin(sel_nationality)
    if search_name:
        mask &= players_full["name"].str.contains(search_name, case=False, na=False)
    if "player_status" in players_full.columns and sel_status:
        mask &= players_full["player_status"].isin(sel_status)

    filtered = players_full[mask].copy()

    # Sort by position order
    filtered = filtered.sort_values("position_sort")

    st.markdown(f"**{len(filtered)}** players matching filters")

    # Display columns: Name, Club, Position, Age, Nationality, Market Value,
    # Season Apps, Season Goals, Season Assists, Career Apps, Career Goals, Career Assists
    display_cols = ["name", "club", "position", "age", "nationality", "value",
                    "season_apps", "season_goals", "season_assists",
                    "total_appearances", "total_goals", "total_assists",
                    "position_sort"]
    avail_cols = [c for c in display_cols if c in filtered.columns]
    display_df = filtered[avail_cols].copy()
    # Sort by position order
    if "position_sort" in display_df.columns:
        display_df = display_df.sort_values("position_sort")
    display_df["value_millions"] = display_df["value"] / 1_000_000
    # Create position label that sorts correctly when clicked:
    # e.g. "01 · Goalkeeper", "02 · Right-Back", etc.
    pos_order_map_label = {p: f"{i+1:02d} · {p}" for i, p in enumerate(POSITION_ORDER)}
    display_df["position_label"] = display_df["position"].map(pos_order_map_label).fillna(display_df["position"])
    show_cols = ["name", "club", "position_label", "age", "nationality", "value_millions",
                 "season_apps", "season_goals", "season_assists",
                 "total_appearances", "total_goals", "total_assists"]
    show_cols = [c for c in show_cols if c in display_df.columns]
    display_df_show = display_df[show_cols].copy()
    col_names = ["Name", "Club", "Position", "Age", "Nationality",
                 "Market Value (€M)", "Season Apps", "Season Goals", "Season Assists",
                 "Career Apps", "Career Goals", "Career Assists"][:len(show_cols)]
    display_df_show.columns = col_names
    display_df_show.index = range(1, len(display_df_show) + 1)

    st.dataframe(
        display_df_show, use_container_width=True, height=420,
        column_config={
            "Market Value (€M)": st.column_config.NumberColumn(format="€%.2fM"),
        },
    )

    # --- Player detail ---
    st.markdown("---")
    st.markdown("### Player Detail Card")
    # Build name (club) list for dropdown
    player_options = sorted(
        filtered[["name", "club"]].drop_duplicates().apply(
            lambda r: player_display_name(r["name"], r["club"]), axis=1
        ).tolist()
    )
    if player_options:
        selected_display = st.selectbox("Select a player", player_options)
        # Extract actual name from "Name (Club)"
        selected_player = selected_display.rsplit(" (", 1)[0] if selected_display else None
        if selected_player:
            p = players_full[players_full["name"] == selected_player].iloc[0]
            # Get current season stats
            cs = get_current_season_stats(seasons, selected_player)

            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("Market Value", fmt_value(p["value"]))
            dc2.metric("Age", int(p["age"]))
            dc3.metric("Position", POS_LABEL_MAP.get(p["position"], p["position"]))
            dc4.metric("Season Appearances", cs["season_apps"])

            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(f"**Club:** {p['club']}")
                st.markdown(f"**Nationality:** {p['nationality']}")
                h = p.get("height")
                if pd.notna(h):
                    st.markdown(f"**Height:** {h}m")
                st.markdown(f"**Foot:** {p.get('foot', '-')}")
                st.markdown(f"**Season 25/26 Goals / Assists:** {cs['season_goals']} / {cs['season_assists']}")
                goals = int(p.get("total_goals", 0))
                assists = int(p.get("total_assists", 0))
                st.markdown(f"**Career Goals / Assists:** {goals} / {assists}")

            with sc2:
                # Value history chart
                pid_matches = value_history[value_history["name"] == selected_player]
                if not pid_matches.empty:
                    fig_vh = px.line(
                        pid_matches,
                        x="date",
                        y="value",
                        labels={"value": "Market Value (\u20AC)", "date": "Date"},
                        title=f"Value History \u2014 {selected_display}",
                        color_discrete_sequence=[COLORS["accent"]],
                    )
                    fig_vh.update_traces(mode="lines+markers", marker=dict(size=4))
                    plotly_defaults(fig_vh)
                    st.plotly_chart(fig_vh, use_container_width=True)
                else:
                    st.info("No value history available for this player.")
    else:
        st.info("No players match the current filters.")


# ===================================================================
# PAGE -- CLUB ANALYSIS
# ===================================================================
elif page == "Club Analysis":
    st.markdown("# Club Analysis")

    mode = st.radio("Mode", ["Single Club", "Compare Two Clubs"], horizontal=True)

    if mode == "Single Club":
        sel_club = st.selectbox("Select Club", sorted(players["club"].unique()))
        squad = players_full[players_full["club"] == sel_club].copy()

        # KPIs
        ck1, ck2, ck3, ck4 = st.columns(4)
        ck1.metric("Squad Size", len(squad))
        ck2.metric("Total Value", fmt_value_short(squad["value"].sum()))
        ck3.metric("Avg Value", fmt_value(squad["value"].mean()))
        ck4.metric("Avg Age", f"{squad['age'].mean():.1f}")

        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown("### Player Values")
            # Order by position: GK first (top of chart), then DEF, MID, FWD (bottom)
            # For horizontal bar, y-axis is bottom-to-top, so reverse: FWD at bottom, GK at top
            squad_sorted = squad.copy()
            squad_sorted["pos_sort"] = squad_sorted["position"].map(
                {p: i for i, p in enumerate(POSITION_ORDER)}
            ).fillna(99)
            # Sort descending so GK appears at top of horizontal bar
            squad_sorted = squad_sorted.sort_values(["pos_sort", "value"], ascending=[False, True])

            fig_bar = px.bar(
                squad_sorted,
                y="name",
                x="value",
                orientation="h",
                color="position_group",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"value": "Market Value (\u20AC)", "name": "Player", "position_group": "Pos Group"},
                height=max(400, len(squad) * 22),
                category_orders={"position_group": POSITION_GROUP_ORDER},
            )
            plotly_defaults(fig_bar)
            fig_bar.update_layout(yaxis_title="", yaxis={"categoryorder": "array",
                                  "categoryarray": squad_sorted["name"].tolist()})
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            st.markdown("### Squad Composition")
            pos_grp = squad.groupby("position_group").size().reset_index(name="count")
            # Order: GK, DEF, MID, FWD
            pos_grp["sort_key"] = pos_grp["position_group"].map(
                {g: i for i, g in enumerate(POSITION_GROUP_ORDER)}
            ).fillna(99)
            pos_grp = pos_grp.sort_values("sort_key")
            fig_comp = px.pie(
                pos_grp,
                values="count",
                names="position_group",
                hole=0.5,
                color_discrete_sequence=px.colors.qualitative.Set2,
                category_orders={"position_group": POSITION_GROUP_ORDER},
            )
            fig_comp.update_traces(textinfo="value+label",
                                    sort=False)  # Keep our custom order
            plotly_defaults(fig_comp)
            fig_comp.update_layout(showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True)

            st.markdown("### Age Distribution")
            fig_age = px.histogram(
                squad,
                x="age",
                nbins=15,
                color_discrete_sequence=[COLORS["secondary"]],
                labels={"age": "Age"},
            )
            fig_age.update_layout(yaxis_title="Players", bargap=0.1)
            plotly_defaults(fig_age)
            st.plotly_chart(fig_age, use_container_width=True)

    else:
        cc1, cc2 = st.columns(2)
        clubs_sorted = sorted(players["club"].unique())
        with cc1:
            club_a = st.selectbox("Club A", clubs_sorted, index=0)
        with cc2:
            club_b = st.selectbox("Club B", clubs_sorted, index=min(1, len(clubs_sorted) - 1))

        sq_a = players_full[players_full["club"] == club_a]
        sq_b = players_full[players_full["club"] == club_b]

        # Side-by-side KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{club_a} \u2014 Total Value", fmt_value_short(sq_a["value"].sum()))
        c2.metric(f"{club_a} \u2014 Avg Age", f"{sq_a['age'].mean():.1f}")
        c3.metric(f"{club_b} \u2014 Total Value", fmt_value_short(sq_b["value"].sum()))
        c4.metric(f"{club_b} \u2014 Avg Age", f"{sq_b['age'].mean():.1f}")

        # Comparison bar chart (position group values)
        def club_pos_summary(df: pd.DataFrame, club_name: str) -> pd.DataFrame:
            grp = df.groupby("position_group").agg(
                total_value=("value", "sum"),
                avg_value=("value", "mean"),
                count=("name", "count"),
            ).reset_index()
            grp["club"] = club_name
            return grp

        comp = pd.concat([club_pos_summary(sq_a, club_a), club_pos_summary(sq_b, club_b)])
        fig_comp = px.bar(
            comp,
            x="position_group",
            y="total_value",
            color="club",
            barmode="group",
            labels={"total_value": "Total Value (\u20AC)", "position_group": "Position Group"},
            title="Squad Value by Position Group",
            color_discrete_sequence=[COLORS["primary"], COLORS["accent"]],
            category_orders={"position_group": POSITION_GROUP_ORDER},
        )
        plotly_defaults(fig_comp)
        st.plotly_chart(fig_comp, use_container_width=True)

        # Top players side by side
        left, right = st.columns(2)
        with left:
            st.markdown(f"### Top 10 \u2014 {club_a}")
            top_a = sq_a.nlargest(10, "value")[["name", "position", "age", "value"]].copy()
            top_a["position"] = top_a["position"].map(POS_LABEL_MAP).fillna(top_a["position"])
            top_a["value"] = top_a["value"] / 1_000_000
            top_a.columns = ["Name", "Position", "Age", "Value (€M)"]
            st.dataframe(top_a, use_container_width=True, hide_index=True,
                         column_config={"Value (€M)": st.column_config.NumberColumn(format="€%.2fM")})
        with right:
            st.markdown(f"### Top 10 \u2014 {club_b}")
            top_b = sq_b.nlargest(10, "value")[["name", "position", "age", "value"]].copy()
            top_b["position"] = top_b["position"].map(POS_LABEL_MAP).fillna(top_b["position"])
            top_b["value"] = top_b["value"] / 1_000_000
            top_b.columns = ["Name", "Position", "Age", "Value (€M)"]
            st.dataframe(top_b, use_container_width=True, hide_index=True,
                         column_config={"Value (€M)": st.column_config.NumberColumn(format="€%.2fM")})


# ===================================================================
# PAGE -- VALUE TRENDS
# ===================================================================
elif page == "Value Trends":
    st.markdown("# Value Trends")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Player Value Over Time",
        "Avg Value by Position",
        "Age vs Value Curves",
        "Top Movers",
    ])

    with tab1:
        st.markdown("### Player Value Over Time")
        # Build display name options
        # Use current club from players, not from value history
        player_club_map = players.drop_duplicates("name").set_index("name")["club"].to_dict()
        name_club_map = {n: player_club_map.get(n, "") for n in value_history["name"].unique()}
        avail_names = sorted(value_history["name"].unique())
        avail_display = [player_display_name(n, name_club_map.get(n, "")) for n in avail_names]
        name_to_display = dict(zip(avail_names, avail_display))
        display_to_name = dict(zip(avail_display, avail_names))

        # Default to top 3 by current value
        top3 = players.nlargest(3, "value")["name"].tolist()
        defaults = [name_to_display[n] for n in top3 if n in name_to_display]
        sel_players_vt = st.multiselect(
            "Select players", avail_display, default=defaults, key="vt_players"
        )
        if sel_players_vt:
            sel_names = [display_to_name[d] for d in sel_players_vt]
            vt_data = value_history[value_history["name"].isin(sel_names)].copy()
            vt_data["display_name"] = vt_data["name"].map(name_to_display)
            fig_vt = px.line(
                vt_data,
                x="date",
                y="value",
                color="display_name",
                labels={"value": "Market Value (\u20AC)", "date": "Date", "display_name": "Player"},
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig_vt.update_traces(mode="lines+markers", marker=dict(size=3))
            plotly_defaults(fig_vt)
            fig_vt.update_layout(height=500)
            st.plotly_chart(fig_vt, use_container_width=True)
        else:
            st.info("Select one or more players to display.")

    with tab2:
        st.markdown("### Average Value by Position Over Time")
        # Merge position info into value_history via player name
        vh_pos = value_history.merge(
            players[["name", "position", "position_group"]].drop_duplicates("name"),
            on="name",
            how="left",
        )
        vh_pos = vh_pos.dropna(subset=["position_group", "date"])
        vh_pos["year"] = vh_pos["date"].dt.year
        # Filter from 2010 only
        vh_pos = vh_pos[vh_pos["year"] >= 2010]
        avg_by_pos = vh_pos.groupby(["year", "position_group"])["value"].mean().reset_index()
        fig_pos = px.line(
            avg_by_pos,
            x="year",
            y="value",
            color="position_group",
            labels={"value": "Avg Value (\u20AC)", "year": "Year", "position_group": "Pos Group"},
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Set2,
            category_orders={"position_group": POSITION_GROUP_ORDER},
        )
        plotly_defaults(fig_pos)
        fig_pos.update_layout(height=480)
        st.plotly_chart(fig_pos, use_container_width=True)

    with tab3:
        st.markdown("### Age vs Market Value by Position")
        pv = players[players["value"] > 0].copy()
        fig_scatter = px.scatter(
            pv,
            x="age",
            y="value",
            color="position_group",
            size="value",
            size_max=20,
            hover_name="name",
            labels={"value": "Market Value (\u20AC)", "age": "Age", "position_group": "Pos Group"},
            color_discrete_sequence=px.colors.qualitative.Set2,
            category_orders={"position_group": POSITION_GROUP_ORDER},
            opacity=0.7,
        )
        # Add trendlines per group
        for grp in POSITION_GROUP_ORDER:
            subset = pv[pv["position_group"] == grp]
            if len(subset) > 5:
                z = np.polyfit(subset["age"], subset["value"], 2)
                poly = np.poly1d(z)
                x_range = np.linspace(subset["age"].min(), subset["age"].max(), 50)
                fig_scatter.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=poly(x_range),
                        mode="lines",
                        name=f"{grp} trend",
                        line=dict(dash="dash", width=2),
                        showlegend=False,
                    )
                )
        plotly_defaults(fig_scatter)
        fig_scatter.update_layout(height=520)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab4:
        st.markdown("### Top Movers This Season (2025/26)")
        st.markdown("Value change from the start of the current season (July 2025) to now.")

        # Get value at start of season (closest to July 2025)
        season_start = pd.Timestamp("2025-07-01")
        vh_season = value_history.dropna(subset=["date"]).copy()

        # For each player, get their value closest to season start (on or after)
        after_start = vh_season[vh_season["date"] >= season_start].sort_values("date")
        start_vals = after_start.groupby("name").first()[["value", "date"]].rename(
            columns={"value": "start_value", "date": "start_date"}
        ).reset_index()

        # Latest value for each player
        latest = vh_season.sort_values("date").groupby("name").tail(1)[["name", "value", "date"]].rename(
            columns={"value": "latest_value", "date": "latest_date"}
        )

        changes = latest.merge(start_vals, on="name", how="inner")
        changes = changes[changes["start_date"] != changes["latest_date"]]  # exclude no-change
        changes["change"] = changes["latest_value"] - changes["start_value"]
        changes["change_millions"] = changes["change"] / 1_000_000
        changes["pct_change"] = (changes["change"] / changes["start_value"].replace(0, np.nan)) * 100

        # Add current club info for display names
        current_club_map = players.drop_duplicates("name").set_index("name")["club"].to_dict()
        changes["current_club"] = changes["name"].map(current_club_map).fillna("")
        changes["display_name"] = changes.apply(
            lambda r: player_display_name(r["name"], r["current_club"]), axis=1
        )

        # Top 15 appreciators - FULL WIDTH, VERTICALLY stacked
        st.markdown("#### Top 15 Biggest Appreciators")
        top_up = changes.nlargest(15, "change").copy()
        fig_up = px.bar(
            top_up.sort_values("change", ascending=True),
            y="display_name",
            x="change_millions",
            orientation="h",
            color_discrete_sequence=[COLORS["positive"]],
            labels={"change_millions": "Value Change (\u20ACM)", "display_name": ""},
            text=top_up.sort_values("change", ascending=True)["change"].apply(lambda v: f"+{fmt_value(v)}"),
        )
        fig_up.update_traces(textposition="outside", textfont_size=10)
        plotly_defaults(fig_up)
        fig_up.update_layout(height=520, xaxis_title="Value Change (\u20ACM)",
                             xaxis=dict(range=[0, top_up["change_millions"].max() * 1.5]))
        st.plotly_chart(fig_up, use_container_width=True)

        # Top 15 depreciators - FULL WIDTH
        st.markdown("#### Top 15 Biggest Depreciators")
        top_down = changes.nsmallest(15, "change").copy()
        fig_down = px.bar(
            top_down.sort_values("change", ascending=False),
            y="display_name",
            x="change_millions",
            orientation="h",
            color_discrete_sequence=[COLORS["negative"]],
            labels={"change_millions": "Value Change (\u20ACM)", "display_name": ""},
            text=top_down.sort_values("change", ascending=False)["change"].apply(lambda v: fmt_value(v)),
        )
        fig_down.update_traces(textposition="outside", textfont_size=10)
        plotly_defaults(fig_down)
        fig_down.update_layout(height=520, xaxis_title="Value Change (\u20ACM)",
                               xaxis=dict(range=[top_down["change_millions"].min() * 1.5, 0]))
        st.plotly_chart(fig_down, use_container_width=True)

        # Search box for individual player season value change
        st.markdown("---")
        st.markdown("### Search Player Season Value Change")
        search_mover = st.text_input(
            "Search player name",
            placeholder="Type a player name to see their season value change...",
            key="mover_search",
        )
        if search_mover:
            matches = changes[changes["name"].str.contains(search_mover, case=False, na=False)]
            if matches.empty:
                st.warning("No matching players found.")
            else:
                for _, row in matches.iterrows():
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric(
                        f"{row['display_name']} - Start of Season",
                        fmt_value(row["start_value"]),
                    )
                    mc2.metric(
                        "Current Value",
                        fmt_value(row["latest_value"]),
                    )
                    change_val = row["change"]
                    mc3.metric(
                        "Change",
                        fmt_value(change_val),
                        delta=f"{row['pct_change']:.1f}%" if pd.notna(row['pct_change']) else None,
                    )


# ===================================================================
# PAGE -- HEAD-TO-HEAD (Predictive Insights removed for rework)
# ===================================================================
elif page == "Head-to-Head":
    st.markdown("# Head-to-Head Comparison")

    # Multiple filters
    filt_c1, filt_c2, filt_c3 = st.columns(3)
    with filt_c1:
        all_positions_h2h = ["All"] + [p for p in POSITION_ORDER if p in players_full["position"].dropna().unique()]
        sel_h2h_pos = st.selectbox("Filter by Position", all_positions_h2h, index=0)
    with filt_c2:
        all_clubs_h2h = ["All"] + sorted(players_full["club"].dropna().unique())
        sel_h2h_club = st.selectbox("Filter by Club", all_clubs_h2h, index=0)
    with filt_c3:
        all_groups_h2h = ["All"] + POSITION_GROUP_ORDER
        sel_h2h_group = st.selectbox("Filter by Position Group", all_groups_h2h, index=0)

    h2h_pool = players_full.copy()
    if sel_h2h_pos != "All":
        h2h_pool = h2h_pool[h2h_pool["all_positions"].apply(lambda ap: sel_h2h_pos in str(ap))]
    if sel_h2h_club != "All":
        h2h_pool = h2h_pool[h2h_pool["club"] == sel_h2h_club]
    if sel_h2h_group != "All":
        h2h_pool = h2h_pool[h2h_pool["position_group"] == sel_h2h_group]

    # Build display names for dropdown
    h2h_display_names = sorted(
        h2h_pool[["name", "club"]].drop_duplicates("name").apply(
            lambda r: player_display_name(r["name"], r["club"]), axis=1
        ).tolist()
    )

    if len(h2h_display_names) < 2:
        st.warning("Not enough players match the selected filters. Please broaden your criteria.")
    else:
        hc1, hc2 = st.columns(2)
        with hc1:
            p1_display = st.selectbox("Player A", h2h_display_names, index=0)
        with hc2:
            default_idx = min(1, len(h2h_display_names) - 1)
            p2_display = st.selectbox("Player B", h2h_display_names, index=default_idx)

        p1_name = p1_display.rsplit(" (", 1)[0]
        p2_name = p2_display.rsplit(" (", 1)[0]

        p1 = players_full[players_full["name"] == p1_name]
        p2 = players_full[players_full["name"] == p2_name]

        if p1.empty or p2.empty:
            st.warning("Could not find one of the selected players.")
        else:
            p1 = p1.iloc[0]
            p2 = p2.iloc[0]

            # Get current season stats
            cs1 = get_current_season_stats(seasons, p1_name)
            cs2 = get_current_season_stats(seasons, p2_name)

            # Side-by-side KPIs
            st.markdown("### At a Glance")
            cols = st.columns([2, 1, 2])
            with cols[0]:
                st.metric("Market Value", fmt_value(p1["value"]))
                st.metric("Age", int(p1["age"]))
                st.metric("Club", p1["club"])
                st.metric("Position", POS_LABEL_MAP.get(p1["position"], p1["position"]))
                st.metric("Season Apps", cs1["season_apps"])
                st.metric("Season Goals / Assists", f"{cs1['season_goals']} / {cs1['season_assists']}")
            with cols[1]:
                st.markdown("<div style='text-align:center; padding-top:60px; font-size:2rem; font-weight:bold; color:#3D5A80;'>VS</div>", unsafe_allow_html=True)
            with cols[2]:
                st.metric("Market Value", fmt_value(p2["value"]))
                st.metric("Age", int(p2["age"]))
                st.metric("Club", p2["club"])
                st.metric("Position", POS_LABEL_MAP.get(p2["position"], p2["position"]))
                st.metric("Season Apps", cs2["season_apps"])
                st.metric("Season Goals / Assists", f"{cs2['season_goals']} / {cs2['season_assists']}")

            # Stats table with both current season and career
            st.markdown("### Detailed Stats Comparison")
            stat_rows = [
                ("Season Appearances", cs1["season_apps"], cs2["season_apps"]),
                ("Season Goals", cs1["season_goals"], cs2["season_goals"]),
                ("Season Assists", cs1["season_assists"], cs2["season_assists"]),
                ("Career Appearances", int(p1.get("total_appearances", 0) or 0), int(p2.get("total_appearances", 0) or 0)),
                ("Career Goals", int(p1.get("total_goals", 0) or 0), int(p2.get("total_goals", 0) or 0)),
                ("Career Assists", int(p1.get("total_assists", 0) or 0), int(p2.get("total_assists", 0) or 0)),
                ("Career Yellow Cards", int(p1.get("total_yellow_cards", 0) or 0), int(p2.get("total_yellow_cards", 0) or 0)),
                ("Career Minutes", int(p1.get("total_minutes", 0) or 0), int(p2.get("total_minutes", 0) or 0)),
                ("Market Value", fmt_value(p1["value"]), fmt_value(p2["value"])),
            ]
            compare_df = pd.DataFrame(stat_rows, columns=["Stat", p1_display, p2_display])
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

            # Value history overlay
            st.markdown("### Value History Overlay")
            vh1 = value_history[value_history["name"] == p1_name]
            vh2 = value_history[value_history["name"] == p2_name]
            if vh1.empty and vh2.empty:
                st.info("No value history available for either player.")
            else:
                fig_overlay = go.Figure()
                if not vh1.empty:
                    fig_overlay.add_trace(go.Scatter(
                        x=vh1["date"],
                        y=vh1["value"],
                        mode="lines+markers",
                        name=p1_display,
                        line=dict(color=COLORS["primary"], width=2),
                        marker=dict(size=4),
                    ))
                if not vh2.empty:
                    fig_overlay.add_trace(go.Scatter(
                        x=vh2["date"],
                        y=vh2["value"],
                        mode="lines+markers",
                        name=p2_display,
                        line=dict(color=COLORS["accent"], width=2),
                        marker=dict(size=4),
                    ))
                fig_overlay.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Market Value (\u20AC)",
                    template="plotly_white",
                    font=dict(family="Inter, sans-serif"),
                    height=450,
                )
                st.plotly_chart(fig_overlay, use_container_width=True)
