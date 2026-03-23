#!/usr/bin/env python3
"""
Unified scraper for multiple league player data from Transfermarkt.
Supports: Top 5 European leagues + MLS, Saudi Pro League, Liga Portugal,
          Eredivisie, Jupiler Pro League.
Scrapes: players, stats, value history, positions, extra stats, league tables.

Usage:
    python3 scrape_top5_leagues.py                  # Scrape all leagues (skip GB1)
    python3 scrape_top5_leagues.py --league ES1     # Scrape La Liga only
    python3 scrape_top5_leagues.py --league MLS1    # Scrape MLS only
    python3 scrape_top5_leagues.py --league SA1     # Scrape Saudi Pro League only
    python3 scrape_top5_leagues.py --resume         # Resume from checkpoint
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import re
import json
import os
import argparse
from pathlib import Path
from datetime import datetime

BASE_URL = "https://www.transfermarkt.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# All supported leagues
LEAGUES = {
    # Top 5 European leagues
    "GB1": {"name": "Premier League", "country": "England",
            "url_slug": "premier-league"},
    "ES1": {"name": "La Liga", "country": "Spain",
            "url_slug": "laliga"},
    "IT1": {"name": "Serie A", "country": "Italy",
            "url_slug": "serie-a"},
    "L1":  {"name": "Bundesliga", "country": "Germany",
            "url_slug": "1-bundesliga"},
    "FR1": {"name": "Ligue 1", "country": "France",
            "url_slug": "ligue-1"},
    # Additional leagues
    "MLS1": {"name": "Major League Soccer", "country": "United States",
             "url_slug": "major-league-soccer"},
    "SA1":  {"name": "Saudi Pro League", "country": "Saudi Arabia",
             "url_slug": "saudi-professional-league"},
    "PO1":  {"name": "Liga Portugal", "country": "Portugal",
             "url_slug": "liga-portugal"},
    "NL1":  {"name": "Eredivisie", "country": "Netherlands",
             "url_slug": "eredivisie"},
    "BE1":  {"name": "Jupiler Pro League", "country": "Belgium",
             "url_slug": "jupiler-pro-league"},
}

DATA_DIR = Path(__file__).parent
CHECKPOINT_FILE = DATA_DIR / "scrape_top5_checkpoint.json"

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def safe_request(url, retries=3, delay=2, timeout=15):
    """Make a request with retries and backoff."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 429:
                wait = (attempt + 1) * 10
                print(f"    Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 403:
                print(f"    Blocked (403), waiting 30s...")
                time.sleep(30)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            print(f"    Timeout on attempt {attempt+1}/{retries}")
            time.sleep(delay * (attempt + 1))
        except requests.exceptions.RequestException as e:
            print(f"    Request error: {e}")
            time.sleep(delay * (attempt + 1))
    return None


def parse_market_value(value_text):
    """Convert market value string like '€80.00m' or '€800k' to numeric."""
    if not value_text or value_text.strip() == "-":
        return None
    text = value_text.strip().replace("€", "").replace(",", ".")
    try:
        if "bn" in text:
            return float(text.replace("bn", "")) * 1_000_000_000
        elif "m" in text:
            return float(text.replace("m", "")) * 1_000_000
        elif "k" in text or "Th." in text:
            text = text.replace("k", "").replace("Th.", "")
            return float(text) * 1_000
        else:
            return float(text)
    except ValueError:
        return None


def extract_player_id(profile_url):
    """Extract numeric player ID from profile URL."""
    m = re.search(r"/spieler/(\d+)", str(profile_url))
    return int(m.group(1)) if m else None


def save_checkpoint(state):
    """Save progress checkpoint."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_checkpoint():
    """Load progress checkpoint."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────
# 1. SCRAPE LEAGUE TABLE
# ─────────────────────────────────────────────────────────────────────

def scrape_league_table(league_code):
    """Scrape current league standings."""
    info = LEAGUES[league_code]
    url = f"{BASE_URL}/{info['url_slug']}/tabelle/wettbewerb/{league_code}"
    print(f"  Fetching league table: {url}")
    resp = safe_request(url)
    if not resp:
        print("  ERROR: Could not fetch league table")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", class_="items")
    if not table:
        print("  WARNING: No table found")
        return pd.DataFrame()

    rows = []
    tbody = table.find("tbody")
    if not tbody:
        return pd.DataFrame()

    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 9:
            continue
        try:
            pos = tds[0].text.strip()
            club_link = tds[2].find("a")
            club = club_link.text.strip() if club_link else tds[2].text.strip()
            matches = tds[3].text.strip()
            wins = tds[4].text.strip()
            draws = tds[5].text.strip()
            losses = tds[6].text.strip()
            goals = tds[7].text.strip()  # "GF:GA" format
            gd = tds[8].text.strip()
            pts = tds[9].text.strip() if len(tds) > 9 else ""

            gf, ga = "", ""
            if ":" in goals:
                parts = goals.split(":")
                gf, ga = parts[0].strip(), parts[1].strip()

            rows.append({
                "position": int(pos) if pos.isdigit() else 0,
                "club": club,
                "matches": int(matches) if matches.isdigit() else 0,
                "wins": int(wins) if wins.isdigit() else 0,
                "draws": int(draws) if draws.isdigit() else 0,
                "losses": int(losses) if losses.isdigit() else 0,
                "goals_for": int(gf) if gf.isdigit() else 0,
                "goals_against": int(ga) if ga.isdigit() else 0,
                "goal_difference": int(gd) if gd.lstrip("-").isdigit() else 0,
                "points": int(pts) if pts.isdigit() else 0,
            })
        except Exception as e:
            continue

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# 2. SCRAPE PLAYERS (SQUAD DATA)
# ─────────────────────────────────────────────────────────────────────

def get_team_links(league_code):
    """Get all team URLs for a given league."""
    info = LEAGUES[league_code]
    url = f"{BASE_URL}/{info['url_slug']}/startseite/wettbewerb/{league_code}"
    print(f"  Fetching teams: {url}")
    resp = safe_request(url)
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    teams = []
    seen = set()

    for link in soup.find_all("a", href=re.compile(r"/startseite/verein/\d+")):
        href = link.get("href", "")
        name = link.text.strip()
        if not name or len(name) < 2:
            name = link.get("title", "")
        if not name or len(name) < 2:
            continue

        team_id = re.search(r"/verein/(\d+)", href)
        if not team_id:
            continue
        tid = team_id.group(1)

        if tid not in seen:
            seen.add(tid)
            teams.append({"name": name, "url": BASE_URL + href, "team_id": tid})

    return teams


def scrape_team_players(team_name, team_url):
    """Scrape all players from a team's squad page."""
    kader_url = team_url.replace("/startseite/", "/kader/")
    if "plus/1" not in kader_url:
        kader_url += "/plus/1"

    resp = safe_request(kader_url)
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    players = []
    table = soup.find("table", class_="items")
    if not table:
        return players

    tbody = table.find("tbody")
    if not tbody:
        return players

    for row in tbody.find_all("tr", class_=["odd", "even"]):
        tds = row.find_all("td")
        if len(tds) < 13:
            continue

        player = {"club": team_name}

        # Shirt number
        player["shirt_number"] = tds[0].text.strip()

        # Name + position from inline table
        inline_table = tds[1].find("table", class_="inline-table")
        if inline_table:
            trs = inline_table.find_all("tr")
            if trs:
                name_link = trs[0].find("a")
                if name_link:
                    player["name"] = name_link.text.strip()
                    player["profile_url"] = BASE_URL + name_link.get("href", "")
            if len(trs) >= 2:
                player["position"] = trs[-1].text.strip()

        if "name" not in player:
            name_link = tds[3].find("a")
            if name_link:
                player["name"] = name_link.text.strip()

        if "position" not in player:
            player["position"] = tds[4].text.strip()

        # DOB + age
        dob_text = tds[5].text.strip()
        dob_match = re.match(r"(\d{2}/\d{2}/\d{4})\s*\((\d+)\)", dob_text)
        if dob_match:
            player["date_of_birth"] = dob_match.group(1)
            player["age"] = int(dob_match.group(2))

        # Nationality
        flags = tds[6].find_all("img")
        nationalities = [f.get("title", "") for f in flags if f.get("title")]
        if nationalities:
            player["nationality"] = ", ".join(nationalities)

        # Height
        height_text = tds[7].text.strip()
        if height_text:
            player["height_m"] = height_text.replace(",", ".").replace("m", "").strip()

        # Foot
        foot_text = tds[8].text.strip().lower()
        if foot_text in ["right", "left", "both"]:
            player["foot"] = foot_text

        # Joined
        player["joined"] = tds[9].text.strip()

        # Signed from
        signed_img = tds[10].find("img")
        if signed_img:
            player["signed_from"] = signed_img.get("title", "")

        # Contract expires
        player["contract_expires"] = tds[11].text.strip()

        # Market value
        mv_text = tds[12].text.strip()
        val = parse_market_value(mv_text)
        if val is not None:
            player["market_value_eur"] = val
            player["market_value_raw"] = mv_text

        if "name" in player:
            players.append(player)

    return players


# ─────────────────────────────────────────────────────────────────────
# 3. SCRAPE CAREER STATS + SEASON-BY-SEASON
# ─────────────────────────────────────────────────────────────────────

def scrape_player_stats(player_id, player_name):
    """Scrape career performance stats for a player."""
    url = f"{BASE_URL}/a/leistungsdaten/spieler/{player_id}/plus/0?saison=ges"
    resp = safe_request(url)
    if not resp:
        return {}, []

    soup = BeautifulSoup(resp.text, "html.parser")
    career = {"player_id": player_id, "name": player_name}
    seasons = []

    # Career totals from the main stats table
    table = soup.find("table", class_="items")
    if table:
        # Total row (tfoot)
        tfoot = table.find("tfoot")
        if tfoot:
            tds = tfoot.find_all("td")
            if len(tds) >= 6:
                career["total_appearances"] = int(tds[2].text.strip() or 0) if tds[2].text.strip().isdigit() else 0
                career["total_goals"] = int(tds[3].text.strip() or 0) if tds[3].text.strip().isdigit() else 0
                career["total_assists"] = int(tds[4].text.strip() or 0) if tds[4].text.strip().isdigit() else 0

        # Per-competition rows
        tbody = table.find("tbody")
        if tbody:
            for tr in tbody.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 6:
                    continue
                comp_link = tds[0].find("a")
                comp = comp_link.text.strip() if comp_link else tds[0].text.strip()
                if not comp:
                    continue
                apps = tds[2].text.strip()
                goals = tds[3].text.strip()
                assists = tds[4].text.strip()
                safe_comp = re.sub(r'[^a-zA-Z0-9 ]', '', comp).strip().replace(' ', '_')
                career[f"{safe_comp}_apps"] = int(apps) if apps.isdigit() else 0
                career[f"{safe_comp}_goals"] = int(goals) if goals.isdigit() else 0
                career[f"{safe_comp}_assists"] = int(assists) if assists.isdigit() else 0

    # Season-by-season detail
    detail_url = f"{BASE_URL}/a/leistungsdatendetails/spieler/{player_id}/plus/0?saison=&verein=&liga=&wettbewerb=&pos=&trainer_id="
    resp2 = safe_request(detail_url)
    if resp2:
        soup2 = BeautifulSoup(resp2.text, "html.parser")
        table2 = soup2.find("table", class_="items")
        if table2:
            tbody2 = table2.find("tbody")
            if tbody2:
                for tr in tbody2.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) < 10:
                        continue
                    try:
                        season_text = tds[0].text.strip()
                        comp_link = tds[1].find("a")
                        comp = comp_link.text.strip() if comp_link else tds[1].text.strip()
                        club_link = tds[2].find("a")
                        club = club_link.text.strip() if club_link else tds[2].text.strip()
                        apps = tds[4].text.strip()
                        goals = tds[5].text.strip()
                        assists = tds[6].text.strip()
                        yellows = tds[7].text.strip()
                        second_y = tds[8].text.strip() if len(tds) > 8 else ""
                        reds = tds[9].text.strip() if len(tds) > 9 else ""
                        mins = tds[10].text.strip().replace(".", "").replace("'", "") if len(tds) > 10 else ""

                        seasons.append({
                            "season": season_text,
                            "competition": comp,
                            "club": club,
                            "appearances": int(apps) if apps.isdigit() else 0,
                            "goals": int(goals) if goals.isdigit() else 0,
                            "assists": int(assists) if assists.isdigit() else 0,
                            "yellow_cards": int(yellows) if yellows.isdigit() else 0,
                            "second_yellows": int(second_y) if second_y.isdigit() else 0,
                            "red_cards": int(reds) if reds.isdigit() else 0,
                            "minutes_played": int(mins) if mins.isdigit() else 0,
                            "player_id": player_id,
                            "name": player_name,
                        })
                    except Exception:
                        continue

    # Also get total minutes, yellows, reds from career stats page
    stats_url = f"{BASE_URL}/a/leistungsdaten/spieler/{player_id}/saison/ges/plus/1"
    resp3 = safe_request(stats_url)
    if resp3:
        soup3 = BeautifulSoup(resp3.text, "html.parser")
        tfoot3 = soup3.find("tfoot")
        if tfoot3:
            tds3 = tfoot3.find_all("td")
            if len(tds3) >= 10:
                career["total_yellow_cards"] = int(tds3[5].text.strip() or 0) if tds3[5].text.strip().isdigit() else 0
                career["total_second_yellows"] = int(tds3[6].text.strip() or 0) if tds3[6].text.strip().isdigit() else 0
                career["total_red_cards"] = int(tds3[7].text.strip() or 0) if tds3[7].text.strip().isdigit() else 0
                mins_text = tds3[8].text.strip().replace(".", "").replace("'", "")
                career["total_minutes"] = int(mins_text) if mins_text.isdigit() else 0

    return career, seasons


# ─────────────────────────────────────────────────────────────────────
# 4. SCRAPE VALUE HISTORY
# ─────────────────────────────────────────────────────────────────────

def scrape_value_history(player_id, player_name):
    """Scrape market value history from Transfermarkt API."""
    url = f"{BASE_URL}/ceapi/marketValueDevelopment/graph/{player_id}"
    resp = safe_request(url, timeout=10)
    if not resp:
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    records = []
    entries = data.get("list", data) if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []

    for entry in entries:
        try:
            records.append({
                "player_id": player_id,
                "name": player_name,
                "date": entry.get("datum_mw", ""),
                "timestamp": entry.get("x", ""),
                "market_value_eur": entry.get("y", 0),
                "market_value_formatted": entry.get("mw", ""),
                "club": entry.get("verein", ""),
                "age": entry.get("age", ""),
            })
        except Exception:
            continue

    return records


# ─────────────────────────────────────────────────────────────────────
# 5. SCRAPE POSITIONS
# ─────────────────────────────────────────────────────────────────────

def scrape_positions(player_id, player_name, profile_url):
    """Scrape main and alternative positions from player profile."""
    resp = safe_request(profile_url)
    if not resp:
        return {"player_id": player_id, "name": player_name, "main_position": "", "all_positions": ""}

    soup = BeautifulSoup(resp.text, "html.parser")
    positions = []

    # Strategy 1: data-header labels
    for li in soup.find_all("li"):
        label = li.find("span", class_="data-header__label")
        if label and "Position:" in label.text:
            content = li.find("span", class_="data-header__content")
            if content:
                positions.append(content.text.strip())

    # Strategy 2: detail-position boxes
    for box in soup.find_all("div", class_="detail-position__box"):
        pos_text = box.text.strip()
        if pos_text and pos_text not in positions:
            positions.append(pos_text)

    # Strategy 3: info table
    for th in soup.find_all("th"):
        if "Position" in th.text or "Hauptposition" in th.text:
            td = th.find_next("td")
            if td:
                pos = td.text.strip()
                if pos and pos not in positions:
                    positions.append(pos)

    main_pos = positions[0] if positions else ""
    all_pos = ", ".join(dict.fromkeys(positions)) if positions else main_pos

    return {
        "player_id": player_id,
        "name": player_name,
        "main_position": main_pos,
        "all_positions": all_pos,
    }


# ─────────────────────────────────────────────────────────────────────
# 6. SCRAPE EXTRA STATS (injuries, transfers, international)
# ─────────────────────────────────────────────────────────────────────

def scrape_extra_stats(player_id, player_name, profile_url):
    """Scrape injury history, transfer history, and international caps."""
    result = {
        "player_id": player_id,
        "name": player_name,
        "international_caps": 0,
        "international_goals": 0,
        "national_team_level": "",
        "career_injuries": 0,
        "career_days_injured": 0,
        "num_transfers": 0,
        "total_transfer_fees_eur": 0,
        "highest_fee_eur": 0,
        "num_clubs": 0,
    }

    # Profile page for international data
    resp = safe_request(profile_url)
    if resp:
        soup = BeautifulSoup(resp.text, "html.parser")
        # Look for national team info
        for a in soup.find_all("a", href=re.compile(r"/nationalmannschaft/")):
            text = a.text.strip()
            if text:
                result["national_team_level"] = text
                break

    # Injuries
    inj_url = f"{BASE_URL}/a/verletzungen/spieler/{player_id}"
    resp = safe_request(inj_url, timeout=15)
    if resp:
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", class_="items")
        if table:
            tbody = table.find("tbody")
            if tbody:
                injuries = tbody.find_all("tr")
                result["career_injuries"] = len(injuries)
                total_days = 0
                for tr in injuries:
                    tds = tr.find_all("td")
                    if len(tds) >= 4:
                        days_text = tds[-1].text.strip().replace(" days", "").replace(" day", "")
                        if days_text.isdigit():
                            total_days += int(days_text)
                result["career_days_injured"] = total_days

    time.sleep(0.5)

    # Transfers
    txf_url = f"{BASE_URL}/a/transfers/spieler/{player_id}"
    resp = safe_request(txf_url, timeout=15)
    if resp:
        soup = BeautifulSoup(resp.text, "html.parser")
        clubs = set()
        fees = []
        for table in soup.find_all("table", class_="items"):
            tbody = table.find("tbody")
            if not tbody:
                continue
            for tr in tbody.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 5:
                    continue
                # Club names
                for a in tr.find_all("a", href=re.compile(r"/verein/")):
                    cname = a.text.strip()
                    if cname and len(cname) > 1:
                        clubs.add(cname)
                # Fee
                fee_text = tds[-1].text.strip() if tds else ""
                fee = parse_market_value(fee_text)
                if fee and fee > 0:
                    fees.append(fee)

        result["num_transfers"] = len(fees)
        result["total_transfer_fees_eur"] = sum(fees)
        result["highest_fee_eur"] = max(fees) if fees else 0
        result["num_clubs"] = len(clubs)

    return result


# ─────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────

def scrape_league(league_code, resume_from=None):
    """Full pipeline for a single league."""
    info = LEAGUES[league_code]
    league_name = info["name"]
    country = info["country"]

    print("\n" + "=" * 70)
    print(f"  SCRAPING: {league_name} ({country}) [{league_code}]")
    print("=" * 70)

    # Create league output directory
    league_dir = DATA_DIR / f"data_{league_code}"
    league_dir.mkdir(exist_ok=True)

    # ── Step 1: League Table ──
    table_path = league_dir / "league_table.csv"
    if not table_path.exists():
        print(f"\n[1/6] League Table")
        lt = scrape_league_table(league_code)
        if not lt.empty:
            lt.to_csv(table_path, index=False)
            print(f"  Saved {len(lt)} rows -> {table_path}")
        time.sleep(2)
    else:
        print(f"\n[1/6] League Table: SKIPPED (already exists)")

    # ── Step 2: Players (squad data) ──
    players_path = league_dir / "players.csv"
    if not players_path.exists():
        print(f"\n[2/6] Squad Data")
        teams = get_team_links(league_code)
        print(f"  Found {len(teams)} teams")

        all_players = []
        for i, team in enumerate(teams, 1):
            print(f"  [{i}/{len(teams)}] {team['name']}")
            try:
                pl = scrape_team_players(team["name"], team["url"])
                all_players.extend(pl)
                print(f"    -> {len(pl)} players")
            except Exception as e:
                print(f"    ERROR: {e}")
            time.sleep(2)

        if all_players:
            df = pd.DataFrame(all_players)
            cols = ["name", "club", "position", "age", "date_of_birth", "nationality",
                    "height_m", "foot", "shirt_number", "market_value_eur",
                    "market_value_raw", "joined", "signed_from", "contract_expires",
                    "profile_url"]
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            df = df[cols]
            df.to_csv(players_path, index=False)
            print(f"  Saved {len(df)} players -> {players_path}")
    else:
        print(f"\n[2/6] Squad Data: SKIPPED (already exists)")

    # Load players for remaining steps
    if not players_path.exists():
        print("  ERROR: No players data. Cannot continue.")
        return
    players_df = pd.read_csv(players_path)
    players_df["player_id"] = players_df["profile_url"].apply(extract_player_id)
    total = len(players_df)
    print(f"\n  Total players to process: {total}")

    # ── Step 3: Career Stats + Seasons ──
    stats_path = league_dir / "player_stats.csv"
    seasons_path = league_dir / "player_seasons.csv"
    if not stats_path.exists():
        print(f"\n[3/6] Career Stats & Season Details")
        all_stats = []
        all_seasons = []
        start_idx = resume_from.get("stats_idx", 0) if resume_from else 0

        for i, (_, row) in enumerate(players_df.iterrows()):
            if i < start_idx:
                continue
            pid = row["player_id"]
            name = row["name"]
            if pd.isna(pid):
                continue
            pid = int(pid)

            print(f"  [{i+1}/{total}] {name} (ID: {pid})")
            try:
                career, szns = scrape_player_stats(pid, name)
                all_stats.append(career)
                all_seasons.extend(szns)
            except Exception as e:
                print(f"    ERROR: {e}")

            # Checkpoint every 25 players
            if (i + 1) % 25 == 0:
                pd.DataFrame(all_stats).to_csv(stats_path, index=False)
                pd.DataFrame(all_seasons).to_csv(seasons_path, index=False)
                save_checkpoint({"league": league_code, "step": "stats", "stats_idx": i + 1})
                print(f"  ── Checkpoint at {i+1} players ──")

            time.sleep(1.5)

        if all_stats:
            pd.DataFrame(all_stats).to_csv(stats_path, index=False)
        if all_seasons:
            pd.DataFrame(all_seasons).to_csv(seasons_path, index=False)
        print(f"  Saved stats -> {stats_path}")
        print(f"  Saved seasons -> {seasons_path}")
    else:
        print(f"\n[3/6] Career Stats: SKIPPED (already exists)")

    # ── Step 4: Value History ──
    vh_path = league_dir / "player_value_history.csv"
    if not vh_path.exists():
        print(f"\n[4/6] Value History")
        all_vh = []
        start_idx = resume_from.get("vh_idx", 0) if resume_from else 0

        for i, (_, row) in enumerate(players_df.iterrows()):
            if i < start_idx:
                continue
            pid = row["player_id"]
            name = row["name"]
            if pd.isna(pid):
                continue
            pid = int(pid)

            records = scrape_value_history(pid, name)
            all_vh.extend(records)

            if (i + 1) % 50 == 0:
                pd.DataFrame(all_vh).to_csv(vh_path, index=False)
                save_checkpoint({"league": league_code, "step": "vh", "vh_idx": i + 1})
                print(f"  ── Checkpoint at {i+1} players ({len(all_vh)} records) ──")

            time.sleep(0.5)

        if all_vh:
            pd.DataFrame(all_vh).to_csv(vh_path, index=False)
        print(f"  Saved {len(all_vh)} value history records -> {vh_path}")
    else:
        print(f"\n[4/6] Value History: SKIPPED (already exists)")

    # ── Step 5: Positions ──
    pos_path = league_dir / "player_positions.csv"
    if not pos_path.exists():
        print(f"\n[5/6] Player Positions")
        all_pos = []
        start_idx = resume_from.get("pos_idx", 0) if resume_from else 0

        for i, (_, row) in enumerate(players_df.iterrows()):
            if i < start_idx:
                continue
            pid = row["player_id"]
            name = row["name"]
            purl = row["profile_url"]
            if pd.isna(pid) or pd.isna(purl):
                continue
            pid = int(pid)

            pos = scrape_positions(pid, name, purl)
            all_pos.append(pos)

            if (i + 1) % 50 == 0:
                pd.DataFrame(all_pos).to_csv(pos_path, index=False)
                save_checkpoint({"league": league_code, "step": "pos", "pos_idx": i + 1})
                print(f"  ── Checkpoint at {i+1} players ──")

            time.sleep(1.5)

        if all_pos:
            pd.DataFrame(all_pos).to_csv(pos_path, index=False)
        print(f"  Saved {len(all_pos)} positions -> {pos_path}")
    else:
        print(f"\n[5/6] Positions: SKIPPED (already exists)")

    # ── Step 6: Extra Stats (injuries, transfers) ──
    extra_path = league_dir / "player_extra_stats.csv"
    if not extra_path.exists():
        print(f"\n[6/6] Extra Stats (injuries, transfers, international)")
        all_extra = []
        start_idx = resume_from.get("extra_idx", 0) if resume_from else 0

        for i, (_, row) in enumerate(players_df.iterrows()):
            if i < start_idx:
                continue
            pid = row["player_id"]
            name = row["name"]
            purl = row["profile_url"]
            if pd.isna(pid) or pd.isna(purl):
                continue
            pid = int(pid)

            print(f"  [{i+1}/{total}] {name}")
            try:
                extra = scrape_extra_stats(pid, name, purl)
                all_extra.append(extra)
            except Exception as e:
                print(f"    ERROR: {e}")

            if (i + 1) % 25 == 0:
                pd.DataFrame(all_extra).to_csv(extra_path, index=False)
                save_checkpoint({"league": league_code, "step": "extra", "extra_idx": i + 1})
                print(f"  ── Checkpoint at {i+1} players ──")

            time.sleep(1.5)

        if all_extra:
            pd.DataFrame(all_extra).to_csv(extra_path, index=False)
        print(f"  Saved {len(all_extra)} extra stats -> {extra_path}")
    else:
        print(f"\n[6/6] Extra Stats: SKIPPED (already exists)")

    # Clean up checkpoint
    if CHECKPOINT_FILE.exists():
        os.remove(CHECKPOINT_FILE)

    print(f"\n{'='*70}")
    print(f"  DONE: {league_name} ({country}) [{league_code}]")
    print(f"  Output directory: {league_dir}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Scrape multiple leagues from Transfermarkt")
    parser.add_argument("--league", type=str, default=None,
                        help="League code to scrape (e.g. ES1, MLS1, SA1). Default: all except GB1.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--include-gb1", action="store_true",
                        help="Also re-scrape Premier League (GB1)")
    parser.add_argument("--all", action="store_true",
                        help="Scrape ALL leagues including GB1")
    args = parser.parse_args()

    print("=" * 70)
    print("  TRANSFERMARKT MULTI-LEAGUE SCRAPER")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Determine which leagues to scrape
    if args.league:
        if args.league not in LEAGUES:
            print(f"ERROR: Unknown league code '{args.league}'. Valid: {list(LEAGUES.keys())}")
            return
        league_codes = [args.league]
    elif getattr(args, "all"):
        league_codes = list(LEAGUES.keys())
    else:
        # Skip Premier League (already done), scrape all others
        league_codes = [k for k in LEAGUES if k != "GB1"]
        if args.include_gb1:
            league_codes = ["GB1"] + league_codes

    print(f"\nLeagues to scrape: {', '.join(league_codes)}")
    for code in league_codes:
        info = LEAGUES[code]
        print(f"  [{code}] {info['name']} ({info['country']})")

    # Resume support
    checkpoint = load_checkpoint() if args.resume else {}

    for code in league_codes:
        resume_from = checkpoint if checkpoint.get("league") == code else None
        scrape_league(code, resume_from=resume_from)

    print("\n" + "=" * 70)
    print("  ALL DONE!")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Summary
    print("\n  Output files:")
    for code in league_codes:
        league_dir = DATA_DIR / f"data_{code}"
        if league_dir.exists():
            files = list(league_dir.glob("*.csv"))
            total_size = sum(f.stat().st_size for f in files)
            print(f"  [{code}] {LEAGUES[code]['name']}: {len(files)} files, {total_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
