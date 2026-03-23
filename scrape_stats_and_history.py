"""
Scrape player performance stats (career totals) and market value history
for all Premier League players from Transfermarkt.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import json
import os

BASE_URL = "https://www.transfermarkt.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# ─────────────────────────────────────────────
# PERFORMANCE STATS
# ─────────────────────────────────────────────

def scrape_player_stats(player_name, profile_url):
    """
    Scrape career performance stats for a single player.
    Returns a dict with career totals and per-competition breakdown.
    """
    # Extract player slug and ID from profile URL
    # e.g. /erling-haaland/profil/spieler/418560
    match = re.search(r"(/[^/]+)/profil/spieler/(\d+)", profile_url)
    if not match:
        return None

    slug = match.group(1)
    player_id = match.group(2)

    # Career stats page (all seasons aggregated by competition)
    stats_url = f"{BASE_URL}{slug}/leistungsdaten/spieler/{player_id}/plus/0?saison=ges"

    try:
        resp = requests.get(stats_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"    ERROR fetching stats for {player_name}: {e}", flush=True)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", class_="items")
    if not table:
        return None

    result = {
        "player_id": player_id,
        "name": player_name,
    }

    # Parse per-competition rows
    competitions = []
    tbody = table.find("tbody")
    if tbody:
        for row in tbody.find_all("tr"):
            tds = row.find_all("td")
            if len(tds) < 9:
                continue
            comp_name = tds[1].text.strip()
            if not comp_name:
                # Try to get from link/image
                link = tds[1].find("a")
                if link:
                    comp_name = link.get("title", "") or link.text.strip()
            competitions.append({
                "competition": comp_name,
                "appearances": parse_int(tds[2].text.strip()),
                "goals": parse_int(tds[3].text.strip()),
                "assists": parse_int(tds[4].text.strip()),
                "yellow_cards": parse_int(tds[5].text.strip()),
                "second_yellows": parse_int(tds[6].text.strip()),
                "red_cards": parse_int(tds[7].text.strip()),
                "minutes_played": parse_minutes(tds[8].text.strip()),
            })

    result["competitions"] = competitions

    # Parse totals from footer
    tfoot = table.find("tfoot")
    if tfoot:
        row = tfoot.find("tr")
        if row:
            tds = row.find_all("td")
            if len(tds) >= 9:
                result["total_appearances"] = parse_int(tds[2].text.strip())
                result["total_goals"] = parse_int(tds[3].text.strip())
                result["total_assists"] = parse_int(tds[4].text.strip())
                result["total_yellow_cards"] = parse_int(tds[5].text.strip())
                result["total_second_yellows"] = parse_int(tds[6].text.strip())
                result["total_red_cards"] = parse_int(tds[7].text.strip())
                result["total_minutes"] = parse_minutes(tds[8].text.strip())

    # Now also get season-by-season breakdown
    detail_url = f"{BASE_URL}{slug}/leistungsdatendetails/spieler/{player_id}/plus/0?saison=&verein=&liga=&wettbewerb=&pos=&trainer_id="
    try:
        resp2 = requests.get(detail_url, headers=HEADERS, timeout=15)
        resp2.raise_for_status()
        soup2 = BeautifulSoup(resp2.text, "html.parser")
        table2 = soup2.find("table", class_="items")
        if table2:
            seasons = []
            tbody2 = table2.find("tbody")
            if tbody2:
                current_season = None
                for row in tbody2.find_all("tr"):
                    tds = row.find_all("td")
                    if len(tds) < 9:
                        continue
                    season_text = tds[0].text.strip()
                    if season_text:
                        current_season = season_text

                    # Competition name from link
                    comp = ""
                    comp_links = tds[1].find_all("a") if len(tds) > 1 else []
                    if comp_links:
                        comp = comp_links[-1].get("title", "") or comp_links[-1].text.strip()
                    if not comp:
                        comp = tds[2].text.strip() if len(tds) > 2 else ""

                    # Club
                    club = ""
                    club_td = tds[3] if len(tds) > 3 else None
                    if club_td:
                        club_img = club_td.find("img")
                        if club_img:
                            club = club_img.get("title", "")
                        if not club:
                            club_link = club_td.find("a")
                            if club_link:
                                club = club_link.get("title", "") or club_link.text.strip()

                    # Cards column: "Y / 2Y / R" format
                    cards_text = tds[7].text.strip() if len(tds) > 7 else ""
                    cards = [parse_int(c.strip()) for c in cards_text.split("/")]
                    while len(cards) < 3:
                        cards.append(0)

                    seasons.append({
                        "season": current_season,
                        "competition": comp,
                        "club": club,
                        "appearances": parse_int(tds[4].text.strip()) if len(tds) > 4 else 0,
                        "goals": parse_int(tds[5].text.strip()) if len(tds) > 5 else 0,
                        "assists": parse_int(tds[6].text.strip()) if len(tds) > 6 else 0,
                        "yellow_cards": cards[0],
                        "second_yellows": cards[1],
                        "red_cards": cards[2],
                        "minutes_played": parse_minutes(tds[8].text.strip()) if len(tds) > 8 else 0,
                    })

            result["seasons"] = seasons
    except Exception as e:
        print(f"    WARNING: Could not fetch season details for {player_name}: {e}", flush=True)
        result["seasons"] = []

    return result


# ─────────────────────────────────────────────
# MARKET VALUE HISTORY
# ─────────────────────────────────────────────

def scrape_value_history(player_name, player_id):
    """Fetch market value history from the Transfermarkt API."""
    url = f"{BASE_URL}/ceapi/marketValueDevelopment/graph/{player_id}"
    headers = {
        **HEADERS,
        "Referer": f"{BASE_URL}/x/marktwertverlauf/spieler/{player_id}",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    ERROR fetching value history for {player_name}: {e}", flush=True)
        return []

    history = []
    for entry in data.get("list", []):
        history.append({
            "player_id": player_id,
            "name": player_name,
            "date": entry.get("datum_mw", ""),
            "timestamp": entry.get("x"),
            "market_value_eur": entry.get("y"),
            "market_value_formatted": entry.get("mw", ""),
            "club": entry.get("verein", ""),
            "age": entry.get("age", ""),
        })

    return history


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def parse_int(text):
    """Parse integer from text, returning 0 for dashes/empty."""
    text = text.strip().replace(".", "").replace("'", "")
    if not text or text == "-":
        return 0
    try:
        return int(text)
    except ValueError:
        return 0


def parse_minutes(text):
    """Parse minutes like \"2.419'\" or \"27.454'\" to int."""
    text = text.strip().replace("'", "").replace(".", "").replace(",", "")
    if not text or text == "-":
        return 0
    try:
        return int(text)
    except ValueError:
        return 0


# ─────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────

def load_checkpoint(path):
    """Load set of already-processed player IDs."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "player_id" in df.columns:
            return set(df["player_id"].astype(str).unique())
    return set()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Scraping Performance Stats & Market Value History")
    print("=" * 60)

    df = pd.read_csv("premier_league_players.csv")
    print(f"Loaded {len(df)} players\n")

    # Extract player IDs
    df["player_id"] = df["profile_url"].str.extract(r"/spieler/(\d+)")

    # ── PART 1: Performance Stats ──
    print("─" * 40)
    print("PART 1: Performance Stats")
    print("─" * 40)

    stats_file = "player_stats.csv"
    seasons_file = "player_seasons.csv"
    done_stats = load_checkpoint(stats_file)
    print(f"Already scraped: {len(done_stats)} players")

    all_stats = []
    all_seasons = []

    # Load existing data if resuming
    if os.path.exists(stats_file):
        all_stats = pd.read_csv(stats_file).to_dict("records")
    if os.path.exists(seasons_file):
        all_seasons = pd.read_csv(seasons_file).to_dict("records")

    remaining = df[~df["player_id"].isin(done_stats)]
    total = len(df)
    done_count = total - len(remaining)

    for idx, (_, row) in enumerate(remaining.iterrows()):
        current = done_count + idx + 1
        print(f"  [{current}/{total}] {row['name']} ({row['club']})")

        result = scrape_player_stats(row["name"], row["profile_url"])
        if result:
            # Flatten career totals
            stat_row = {
                "player_id": result["player_id"],
                "name": result["name"],
                "total_appearances": result.get("total_appearances", 0),
                "total_goals": result.get("total_goals", 0),
                "total_assists": result.get("total_assists", 0),
                "total_yellow_cards": result.get("total_yellow_cards", 0),
                "total_second_yellows": result.get("total_second_yellows", 0),
                "total_red_cards": result.get("total_red_cards", 0),
                "total_minutes": result.get("total_minutes", 0),
            }

            # Add per-competition totals as columns
            for comp in result.get("competitions", []):
                prefix = comp["competition"][:20].replace(" ", "_").lower()
                stat_row[f"{prefix}_apps"] = comp["appearances"]
                stat_row[f"{prefix}_goals"] = comp["goals"]
                stat_row[f"{prefix}_assists"] = comp["assists"]

            all_stats.append(stat_row)

            # Season-by-season rows
            for s in result.get("seasons", []):
                s["player_id"] = result["player_id"]
                s["name"] = result["name"]
                all_seasons.append(s)

        # Save checkpoint every 20 players
        if (current) % 20 == 0:
            pd.DataFrame(all_stats).to_csv(stats_file, index=False)
            pd.DataFrame(all_seasons).to_csv(seasons_file, index=False)
            print(f"    [checkpoint saved: {len(all_stats)} players]")

        # Be polite — 2 requests per player (stats + detail), so wait between players
        time.sleep(1.5)

    # Final save
    pd.DataFrame(all_stats).to_csv(stats_file, index=False)
    pd.DataFrame(all_seasons).to_csv(seasons_file, index=False)
    print(f"\nSaved {len(all_stats)} player stats to {stats_file}")
    print(f"Saved {len(all_seasons)} season records to {seasons_file}")

    # ── PART 2: Market Value History ──
    print("\n" + "─" * 40)
    print("PART 2: Market Value History")
    print("─" * 40)

    history_file = "player_value_history.csv"
    done_history = load_checkpoint(history_file)
    print(f"Already scraped: {len(done_history)} players")

    all_history = []
    if os.path.exists(history_file):
        all_history = pd.read_csv(history_file).to_dict("records")

    remaining_h = df[~df["player_id"].isin(done_history)]
    done_count_h = total - len(remaining_h)

    for idx, (_, row) in enumerate(remaining_h.iterrows()):
        current = done_count_h + idx + 1
        print(f"  [{current}/{total}] {row['name']}")

        history = scrape_value_history(row["name"], row["player_id"])
        all_history.extend(history)

        # Checkpoint every 50 players
        if current % 50 == 0:
            pd.DataFrame(all_history).to_csv(history_file, index=False)
            print(f"    [checkpoint saved: {len(all_history)} records]")

        time.sleep(0.5)

    pd.DataFrame(all_history).to_csv(history_file, index=False)
    print(f"\nSaved {len(all_history)} value history records to {history_file}")

    # ── SUMMARY ──
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  player_stats.csv        - Career totals for {len(all_stats)} players")
    print(f"  player_seasons.csv      - {len(all_seasons)} season-by-season records")
    print(f"  player_value_history.csv - {len(all_history)} market value data points")
    print("=" * 60)


if __name__ == "__main__":
    main()
