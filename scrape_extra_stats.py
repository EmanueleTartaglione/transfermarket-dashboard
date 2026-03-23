#!/usr/bin/env python3
"""
Scrape additional player attributes from Transfermarkt profiles for richer
regression features.  Reads premier_league_players.csv, visits each profile
page + transfer history page, and extracts:

  - international caps & goals
  - number of career injuries & total days missed
  - contract remaining days
  - number of clubs played for
  - transfer history (number of transfers, total fees paid, highest fee)
  - national team level (full international, U21, U20, etc.)
  - agent info (encoded later)
  - career clean sheets (for GKs)
  - social media following (if available)
  - current season stats (goals, assists, minutes in PL only)

Saves to player_extra_stats.csv with checkpoints every 25 players.
"""

import csv
import json
import os
import re
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime, date

# ── Configuration ──────────────────────────────────────────────────
BASE = "/Users/emanueletartaglione/Desktop/Transfermarket Project"
INPUT_CSV = os.path.join(BASE, "premier_league_players.csv")
OUTPUT_CSV = os.path.join(BASE, "player_extra_stats.csv")
CHECKPOINT_FILE = os.path.join(BASE, "scrape_extra_checkpoint.json")
DELAY = 1.5
CHECKPOINT_INTERVAL = 25

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

FIELD_NAMES = [
    "player_id", "name",
    # International
    "international_caps", "international_goals", "national_team_level",
    # Injuries
    "career_injuries", "career_days_injured",
    # Contract
    "contract_remaining_days",
    # Transfer history
    "num_transfers", "total_transfer_fees_eur", "highest_fee_eur",
    "num_clubs",
    # Current season PL stats
    "pl_season_apps", "pl_season_goals", "pl_season_assists",
    "pl_season_minutes", "pl_season_yellows", "pl_season_reds",
    # Profile extras
    "height_cm", "foot",
    "years_at_current_club",
    # Social / popularity proxy
    "page_views",
]


def extract_player_id(url: str) -> str:
    m = re.search(r"/spieler/(\d+)", url)
    return m.group(1) if m else ""


def safe_int(text: str) -> int:
    """Parse an integer from text, stripping non-digit chars."""
    if not text:
        return 0
    cleaned = re.sub(r"[^\d]", "", text.strip())
    return int(cleaned) if cleaned else 0


def parse_euro_value(raw: str) -> int:
    """Parse Transfermarkt value strings like '€25.00m', '€800k', 'free transfer'."""
    if not raw:
        return 0
    raw = raw.strip().lower().replace(",", ".")
    if "free" in raw or "loan" in raw or "-" == raw or raw == "?":
        return 0
    m = re.search(r"([\d.]+)\s*(m|k|bn)?", raw)
    if not m:
        return 0
    val = float(m.group(1))
    unit = m.group(2) or ""
    if unit == "bn":
        return int(val * 1_000_000_000)
    elif unit == "m":
        return int(val * 1_000_000)
    elif unit == "k":
        return int(val * 1_000)
    return int(val)


# ── Scraping functions ─────────────────────────────────────────────

def scrape_profile(url: str, session: requests.Session) -> dict:
    """Scrape extra data from main profile page."""
    data = {}
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"  ERROR profile {url}: {e}")
        return data

    # ── International caps ─────────────────────────────────────────
    # Look for national team stats box
    int_caps = 0
    int_goals = 0
    nt_level = ""

    # National team section
    nt_section = soup.find("div", {"data-viewport": "Nationalmannschaft"})
    if not nt_section:
        nt_section = soup.find("div", class_=re.compile(r"nationalspieler"))
    if nt_section:
        rows = nt_section.find_all("tr")
        for row in rows:
            tds = row.find_all("td")
            if len(tds) >= 3:
                team_name = tds[0].get_text(strip=True)
                caps_text = tds[1].get_text(strip=True)
                goals_text = tds[2].get_text(strip=True) if len(tds) > 2 else "0"
                caps_val = safe_int(caps_text)
                goals_val = safe_int(goals_text)
                # Take the senior team (most caps, or first row)
                if caps_val > int_caps:
                    int_caps = caps_val
                    int_goals = goals_val
                    nt_level = team_name

    # Fallback: look for "International" in info table
    if int_caps == 0:
        for span in soup.find_all("span", class_="info-table__content--bold"):
            parent_label = span.find_previous("span", class_="info-table__content--regular")
            if parent_label and "cap" in parent_label.get_text(strip=True).lower():
                int_caps = safe_int(span.get_text(strip=True))

    data["international_caps"] = int_caps
    data["international_goals"] = int_goals
    data["national_team_level"] = nt_level

    # ── Height ─────────────────────────────────────────────────────
    height_cm = 0
    for li in soup.find_all("li"):
        label = li.find("span", class_="info-table__content--regular")
        value = li.find("span", class_="info-table__content--bold")
        if label and value:
            label_txt = label.get_text(strip=True).lower()
            value_txt = value.get_text(strip=True)
            if "height" in label_txt:
                # "1,85 m" or "1.85m"
                h = re.search(r"(\d)[,.](\d{2})", value_txt)
                if h:
                    height_cm = int(h.group(1)) * 100 + int(h.group(2))
            elif "foot" in label_txt:
                data["foot"] = value_txt.lower()
            elif "joined" in label_txt:
                try:
                    joined_date = datetime.strptime(value_txt, "%b %d, %Y").date()
                    days_at_club = (date.today() - joined_date).days
                    data["years_at_current_club"] = round(days_at_club / 365.25, 2)
                except:
                    data["years_at_current_club"] = 0
            elif "contract" in label_txt and "expires" in label_txt:
                try:
                    exp_date = datetime.strptime(value_txt, "%b %d, %Y").date()
                    data["contract_remaining_days"] = max(0, (exp_date - date.today()).days)
                except:
                    data["contract_remaining_days"] = 0

    data["height_cm"] = height_cm

    # ── Page views (popularity proxy) ──────────────────────────────
    pv = soup.find("span", class_=re.compile(r"page-views|profile-views"))
    if pv:
        data["page_views"] = safe_int(pv.get_text(strip=True))

    return data


def scrape_injuries(player_id: str, session: requests.Session) -> dict:
    """Scrape injury history page."""
    data = {"career_injuries": 0, "career_days_injured": 0}
    url = f"https://www.transfermarkt.com/a/verletzungen/spieler/{player_id}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"  ERROR injuries {player_id}: {e}")
        return data

    # Count injury rows
    table = soup.find("table", class_="items")
    if table:
        rows = table.find("tbody")
        if rows:
            injury_rows = rows.find_all("tr", class_=re.compile(r"odd|even"))
            data["career_injuries"] = len(injury_rows)
            total_days = 0
            for row in injury_rows:
                tds = row.find_all("td")
                # Last or second-to-last column usually has "X days"
                for td in reversed(tds):
                    txt = td.get_text(strip=True)
                    d = re.search(r"(\d+)\s*day", txt, re.I)
                    if d:
                        total_days += int(d.group(1))
                        break
            data["career_days_injured"] = total_days

    return data


def scrape_transfers(player_id: str, session: requests.Session) -> dict:
    """Scrape transfer history page."""
    data = {"num_transfers": 0, "total_transfer_fees_eur": 0, "highest_fee_eur": 0, "num_clubs": 0}
    url = f"https://www.transfermarkt.com/a/transfers/spieler/{player_id}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"  ERROR transfers {player_id}: {e}")
        return data

    # Transfer rows
    clubs_seen = set()
    total_fees = 0
    highest_fee = 0
    n_transfers = 0

    tables = soup.find_all("table", class_="items")
    for table in tables:
        tbody = table.find("tbody")
        if not tbody:
            continue
        for row in tbody.find_all("tr", class_=re.compile(r"odd|even")):
            n_transfers += 1
            tds = row.find_all("td")
            # Try to find club names
            for td in tds:
                club_links = td.find_all("a", href=re.compile(r"/verein/"))
                for cl in club_links:
                    clubs_seen.add(cl.get_text(strip=True))
            # Fee column is typically the last one
            if tds:
                fee_text = tds[-1].get_text(strip=True)
                fee = parse_euro_value(fee_text)
                total_fees += fee
                if fee > highest_fee:
                    highest_fee = fee

    data["num_transfers"] = n_transfers
    data["total_transfer_fees_eur"] = total_fees
    data["highest_fee_eur"] = highest_fee
    data["num_clubs"] = len(clubs_seen) if clubs_seen else 1

    return data


def scrape_pl_season_stats(player_id: str, session: requests.Session) -> dict:
    """Scrape current Premier League season stats from performance data page."""
    data = {
        "pl_season_apps": 0, "pl_season_goals": 0, "pl_season_assists": 0,
        "pl_season_minutes": 0, "pl_season_yellows": 0, "pl_season_reds": 0,
    }
    url = f"https://www.transfermarkt.com/a/leistungsdaten/spieler/{player_id}/plus/0?saison=2025"
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"  ERROR season stats {player_id}: {e}")
        return data

    # Find Premier League row in the table
    table = soup.find("table", class_="items")
    if not table:
        return data

    tbody = table.find("tbody")
    if not tbody:
        return data

    for row in tbody.find_all("tr"):
        tds = row.find_all("td")
        row_text = row.get_text(strip=True).lower()
        if "premier league" in row_text or "premier-league" in row_text:
            # Parse stats from columns
            nums = []
            for td in tds:
                txt = td.get_text(strip=True).replace(".", "").replace("'", "")
                nums.append(safe_int(txt))
            # Typical column order: competition, apps, goals, assists, yellows, 2nd yellow, reds, minutes
            if len(nums) >= 4:
                data["pl_season_apps"] = nums[1] if len(nums) > 1 else 0
                data["pl_season_goals"] = nums[2] if len(nums) > 2 else 0
                data["pl_season_assists"] = nums[3] if len(nums) > 3 else 0
                if len(nums) > 4:
                    data["pl_season_yellows"] = nums[4]
                if len(nums) > 6:
                    data["pl_season_reds"] = nums[6]
                if len(nums) > 7:
                    data["pl_season_minutes"] = nums[-1]
            break

    return data


# ── Main scraper ───────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed_ids": [], "results": []}


def save_checkpoint(ckpt: dict):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(ckpt, f)


def save_csv(results: list):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main():
    # Load players
    players = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            players.append(row)
    print(f"Loaded {len(players)} players from {INPUT_CSV}")

    ckpt = load_checkpoint()
    completed = set(ckpt.get("completed_ids", []))
    results = ckpt.get("results", [])

    remaining = [(i, p) for i, p in enumerate(players)
                 if extract_player_id(p.get("profile_url", "")) not in completed]
    print(f"Already completed: {len(completed)}, remaining: {len(remaining)}")

    session = requests.Session()

    for idx, (orig_idx, player) in enumerate(remaining):
        name = player["name"]
        profile_url = player.get("profile_url", "")
        player_id = extract_player_id(profile_url)
        if not player_id:
            print(f"  [{orig_idx+1}] Skipping {name} - no profile URL")
            continue

        print(f"  [{orig_idx+1}/{len(players)}] {name} (ID: {player_id})")

        row = {"player_id": player_id, "name": name}

        # 1. Profile page
        profile_data = scrape_profile(profile_url, session)
        row.update(profile_data)
        time.sleep(DELAY)

        # 2. Injury history
        injury_data = scrape_injuries(player_id, session)
        row.update(injury_data)
        time.sleep(DELAY)

        # 3. Transfer history
        transfer_data = scrape_transfers(player_id, session)
        row.update(transfer_data)
        time.sleep(DELAY)

        # 4. Current PL season stats
        season_data = scrape_pl_season_stats(player_id, session)
        row.update(season_data)
        time.sleep(DELAY)

        results.append(row)
        completed.add(player_id)

        # Checkpoint
        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            print(f"  ── Checkpoint at {len(completed)} players ──")
            ckpt["completed_ids"] = list(completed)
            ckpt["results"] = results
            save_checkpoint(ckpt)
            save_csv(results)

    # Final save
    ckpt["completed_ids"] = list(completed)
    ckpt["results"] = results
    save_checkpoint(ckpt)
    save_csv(results)
    print(f"\nDone! Saved {len(results)} players to {OUTPUT_CSV}")

    # Clean up checkpoint file
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint file removed.")


if __name__ == "__main__":
    main()
