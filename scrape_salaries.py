#!/usr/bin/env python3
"""
Scrape Premier League player salary/wage data from Capology.
Extracts weekly and annual wages for all PL players.

Saves to player_salaries.csv with checkpoints every 25 players.
"""

import csv
import json
import os
import re
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────
BASE = "/Users/emanueletartaglione/Desktop/Transfermarket Project"
OUTPUT_CSV = os.path.join(BASE, "player_salaries.csv")
CHECKPOINT_FILE = os.path.join(BASE, "scrape_salaries_checkpoint.json")
DELAY = 2.0
CHECKPOINT_INTERVAL = 25

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

FIELD_NAMES = [
    "name",
    "club",
    "weekly_wage_eur",
    "annual_wage_eur",
]

# URL patterns to try, in order of preference.
# Capology organises leagues by country code and season.
LEAGUE_URL_PATTERNS = [
    "https://www.capology.com/uk/premier-league/salaries/2025-26/",
    "https://www.capology.com/uk/premier-league/salaries/2024-25/",
    "https://www.capology.com/uk/premier-league/salaries/",
]

# Per-club URLs as fallback (club slug -> URL).
CLUB_SLUGS = [
    "arsenal", "aston-villa", "bournemouth", "brentford", "brighton",
    "chelsea", "crystal-palace", "everton", "fulham", "ipswich-town",
    "leicester-city", "liverpool", "manchester-city", "manchester-united",
    "newcastle-united", "nottingham-forest", "southampton",
    "tottenham-hotspur", "west-ham-united", "wolverhampton-wanderers",
]

SEASON_SLUGS = ["2025-26", "2024-25"]


# ── Helpers ────────────────────────────────────────────────────────

def parse_wage(text: str):
    """Parse a wage string like '£120,000' or '€5,200,000' into an integer."""
    if not text:
        return None
    cleaned = text.strip().replace(",", "").replace(" ", "")
    # Remove currency symbols
    cleaned = re.sub(r"[£€$]", "", cleaned)
    # Handle "p/w" or "/wk" suffixes
    cleaned = re.sub(r"(p/w|/wk|/week|/yr|/year|p\.w\.|per\s*week|per\s*year)", "", cleaned, flags=re.I)
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def parse_wage_with_multiplier(text: str):
    """Parse wages that might use K or M shorthand, e.g. '€120K', '€5.2M'."""
    if not text:
        return None
    cleaned = text.strip().replace(",", "").replace(" ", "")
    cleaned = re.sub(r"[£€$]", "", cleaned)
    cleaned = re.sub(r"(p/w|/wk|/week|/yr|/year|p\.w\.|per\s*week|per\s*year)", "", cleaned, flags=re.I)
    cleaned = cleaned.strip().upper()
    if not cleaned:
        return None
    try:
        if cleaned.endswith("M"):
            return int(float(cleaned[:-1]) * 1_000_000)
        elif cleaned.endswith("K"):
            return int(float(cleaned[:-1]) * 1_000)
        else:
            return int(float(cleaned))
    except ValueError:
        return None


def fetch_page(url: str, session: requests.Session):
    """Fetch a page with retries and polite delays."""
    for attempt in range(3):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 403:
                print(f"    [!] 403 Forbidden on attempt {attempt + 1} for {url}")
                time.sleep(5 * (attempt + 1))
            elif resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    [!] Rate-limited (429). Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [!] HTTP {resp.status_code} on attempt {attempt + 1} for {url}")
                time.sleep(3)
        except requests.RequestException as e:
            print(f"    [!] Request error on attempt {attempt + 1}: {e}")
            time.sleep(3)
    return None


# ── Checkpoint system ──────────────────────────────────────────────

def load_checkpoint() -> dict:
    """Load checkpoint data from disk."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"completed_clubs": [], "players": []}


def save_checkpoint(data: dict):
    """Save checkpoint data to disk."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def save_csv(players: list[dict]):
    """Write all collected player records to CSV."""
    if not players:
        return
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(players)
    print(f"  Saved {len(players)} players to {OUTPUT_CSV}")


# ── Scraping strategies ───────────────────────────────────────────

def scrape_league_page(session: requests.Session) -> list[dict]:
    """
    Try to scrape the full league salary page from Capology.
    Returns a list of player dicts, or empty list on failure.
    """
    for url in LEAGUE_URL_PATTERNS:
        print(f"\n  Trying league URL: {url}")
        resp = fetch_page(url, session)
        if resp is None:
            print(f"    Failed to fetch {url}")
            continue

        players = parse_salary_table(resp.text)
        if players:
            print(f"    SUCCESS: Found {len(players)} players from league page")
            return players
        else:
            print(f"    No player data found in page (may be JS-rendered)")

    return []


def scrape_club_pages(session: requests.Session, checkpoint: dict) -> list[dict]:
    """
    Fallback: scrape each club's salary page individually.
    Respects checkpoint to skip already-scraped clubs.
    """
    completed = set(checkpoint.get("completed_clubs", []))
    all_players = list(checkpoint.get("players", []))
    count = 0

    for slug in CLUB_SLUGS:
        if slug in completed:
            print(f"  [skip] {slug} (already in checkpoint)")
            continue

        club_players = []
        found = False

        for season in SEASON_SLUGS:
            url = f"https://www.capology.com/club/{slug}/salaries/{season}/"
            print(f"\n  Trying club URL: {url}")
            resp = fetch_page(url, session)
            if resp is None:
                continue

            club_players = parse_salary_table(resp.text, default_club=slug_to_club_name(slug))
            if club_players:
                print(f"    Found {len(club_players)} players for {slug}")
                found = True
                break
            else:
                print(f"    No data found for {slug} season {season}")

        if found:
            all_players.extend(club_players)
        else:
            print(f"    WARNING: Could not get salary data for {slug}")

        completed.add(slug)
        count += 1

        # Checkpoint periodically
        if count % 5 == 0:
            checkpoint["completed_clubs"] = list(completed)
            checkpoint["players"] = all_players
            save_checkpoint(checkpoint)
            save_csv(all_players)
            print(f"  [checkpoint] {len(all_players)} players saved so far")

        time.sleep(DELAY)

    # Final save
    checkpoint["completed_clubs"] = list(completed)
    checkpoint["players"] = all_players
    save_checkpoint(checkpoint)

    return all_players


def parse_salary_table(html: str, default_club: str = "") -> list[dict]:
    """
    Parse a Capology salary table from raw HTML.
    Works for both league-wide and per-club pages.
    Returns a list of player dicts.
    """
    soup = BeautifulSoup(html, "html.parser")
    players = []

    # Strategy 1: Look for the main salary table by id or class
    table = soup.find("table", {"id": "table"})
    if not table:
        table = soup.find("table", class_="table")
    if not table:
        # Try finding any table with salary-like headers
        for t in soup.find_all("table"):
            header_text = t.get_text(separator=" ").lower()
            if "wage" in header_text or "salary" in header_text or "annual" in header_text:
                table = t
                break

    if not table:
        # Strategy 2: Look for structured div-based layout (some Capology pages)
        return parse_salary_divs(soup, default_club)

    # Determine column indices from headers
    thead = table.find("thead")
    col_map = {}
    if thead:
        headers = thead.find_all("th")
        for i, th in enumerate(headers):
            text = th.get_text(separator=" ").strip().lower()
            if "player" in text or "name" in text:
                col_map["name"] = i
            elif "club" in text or "team" in text:
                col_map["club"] = i
            elif "weekly" in text or "week" in text or "p/w" in text:
                col_map["weekly"] = i
            elif "annual" in text or "yearly" in text or "year" in text:
                col_map["annual"] = i
            elif "gross" in text and "annual" in text:
                col_map["annual"] = i
            elif "gross" in text and "week" in text:
                col_map["weekly"] = i

    # If we could not detect columns by header, try positional guessing
    # Typical Capology layout: Player | Club | Weekly Gross | Annual Gross | ...
    tbody = table.find("tbody")
    if not tbody:
        tbody = table

    rows = tbody.find_all("tr")
    for row in rows:
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue

        player = {}

        # Extract name
        name_idx = col_map.get("name", 0)
        if name_idx < len(cells):
            name_cell = cells[name_idx]
            link = name_cell.find("a")
            player["name"] = (link.text.strip() if link else name_cell.get_text(strip=True))

        # Extract club
        club_idx = col_map.get("club")
        if club_idx is not None and club_idx < len(cells):
            club_cell = cells[club_idx]
            link = club_cell.find("a")
            player["club"] = (link.text.strip() if link else club_cell.get_text(strip=True))
        elif default_club:
            player["club"] = default_club

        # Extract weekly wage
        weekly_idx = col_map.get("weekly")
        if weekly_idx is not None and weekly_idx < len(cells):
            raw = cells[weekly_idx].get_text(strip=True)
            wage = parse_wage(raw) or parse_wage_with_multiplier(raw)
            if wage is not None:
                player["weekly_wage_eur"] = wage

        # Extract annual wage
        annual_idx = col_map.get("annual")
        if annual_idx is not None and annual_idx < len(cells):
            raw = cells[annual_idx].get_text(strip=True)
            wage = parse_wage(raw) or parse_wage_with_multiplier(raw)
            if wage is not None:
                player["annual_wage_eur"] = wage

        # If we got weekly but not annual, compute it (and vice versa)
        if "weekly_wage_eur" in player and "annual_wage_eur" not in player:
            player["annual_wage_eur"] = player["weekly_wage_eur"] * 52
        elif "annual_wage_eur" in player and "weekly_wage_eur" not in player:
            player["weekly_wage_eur"] = player["annual_wage_eur"] // 52

        # Only keep rows that look like actual player data
        if player.get("name") and (player.get("weekly_wage_eur") or player.get("annual_wage_eur")):
            players.append(player)

    return players


def parse_salary_divs(soup: BeautifulSoup, default_club: str = "") -> list[dict]:
    """
    Fallback parser for div-based layouts that some Capology pages use
    instead of standard HTML tables.
    """
    players = []

    # Look for player rows in div-based structure
    player_rows = soup.find_all("div", class_=re.compile(r"player|salary-row|row", re.I))
    for row in player_rows:
        text = row.get_text(separator="|").strip()
        # Skip headers or non-data rows
        if not text or "player" in text.lower() and "wage" in text.lower():
            continue

        parts = [p.strip() for p in text.split("|") if p.strip()]
        if len(parts) < 2:
            continue

        player = {}
        # Try to identify name and wages from the parts
        for part in parts:
            cleaned = part.strip()
            if re.search(r"[£€$]\s*[\d,]+", cleaned):
                wage = parse_wage(cleaned) or parse_wage_with_multiplier(cleaned)
                if wage and wage > 100_000:
                    # Likely annual
                    player.setdefault("annual_wage_eur", wage)
                elif wage:
                    player.setdefault("weekly_wage_eur", wage)
            elif not player.get("name") and re.match(r"^[A-Z]", cleaned) and len(cleaned) > 2:
                # Heuristic: first capitalized string is likely the player name
                if not re.search(r"\d", cleaned):
                    player["name"] = cleaned

        if default_club:
            player["club"] = default_club

        if "weekly_wage_eur" in player and "annual_wage_eur" not in player:
            player["annual_wage_eur"] = player["weekly_wage_eur"] * 52
        elif "annual_wage_eur" in player and "weekly_wage_eur" not in player:
            player["weekly_wage_eur"] = player["annual_wage_eur"] // 52

        if player.get("name") and (player.get("weekly_wage_eur") or player.get("annual_wage_eur")):
            players.append(player)

    return players


def slug_to_club_name(slug: str) -> str:
    """Convert a URL slug like 'manchester-united' to 'Manchester United'."""
    return slug.replace("-", " ").title()


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Capology Premier League Salary Scraper")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    session = requests.Session()
    checkpoint = load_checkpoint()
    all_players = []

    # Strategy 1: Try the full league salary page
    print("\n[Strategy 1] Attempting full league salary page...")
    all_players = scrape_league_page(session)

    if all_players:
        print(f"\nLeague-wide scrape successful: {len(all_players)} players")
    else:
        # Strategy 2: Scrape each club individually
        print("\n[Strategy 2] Falling back to per-club scraping...")
        all_players = scrape_club_pages(session, checkpoint)

    # Save final results
    if all_players:
        # Deduplicate by player name + club
        seen = set()
        unique = []
        for p in all_players:
            key = (p.get("name", "").lower(), p.get("club", "").lower())
            if key not in seen:
                seen.add(key)
                unique.append(p)
        all_players = unique

        save_csv(all_players)

        print("\n" + "=" * 60)
        print(f"SUCCESS! Collected salary data for {len(all_players)} players")
        print(f"Saved to: {OUTPUT_CSV}")
        print("=" * 60)

        # Quick summary: top 10 highest paid
        by_wage = sorted(all_players, key=lambda p: p.get("weekly_wage_eur", 0), reverse=True)
        print("\nTop 10 highest weekly wages:")
        for i, p in enumerate(by_wage[:10], 1):
            weekly = p.get("weekly_wage_eur", "N/A")
            wage_str = f"EUR {weekly:,}" if isinstance(weekly, (int, float)) else str(weekly)
            print(f"  {i:2d}. {p.get('name', 'Unknown'):30s} {p.get('club', ''):25s} {wage_str}")
    else:
        print("\n" + "=" * 60)
        print("WARNING: No salary data collected.")
        print("Capology may be blocking automated requests or the page")
        print("structure may have changed. Consider:")
        print("  1. Using a browser to verify the page loads correctly")
        print("  2. Checking if Capology now requires JavaScript rendering")
        print("  3. Using selenium/playwright as an alternative approach")
        print("=" * 60)

    # Clean up checkpoint on full success
    if all_players and len(all_players) > 100:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("\nCheckpoint file cleaned up (full scrape complete)")


if __name__ == "__main__":
    main()
