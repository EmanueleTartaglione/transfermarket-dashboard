#!/usr/bin/env python3
"""
Scrape Transfermarkt player profiles to get main + secondary positions.
Reads premier_league_players.csv, visits each profile URL, extracts position data,
and saves to player_positions.csv with checkpoints every 50 players.
"""

import csv
import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup

# Configuration
INPUT_CSV = "/Users/emanueletartaglione/Desktop/Transfermarket Project/premier_league_players.csv"
OUTPUT_CSV = "/Users/emanueletartaglione/Desktop/Transfermarket Project/player_positions.csv"
CHECKPOINT_FILE = "/Users/emanueletartaglione/Desktop/Transfermarket Project/scrape_checkpoint.json"
DELAY = 1.5  # seconds between requests
CHECKPOINT_INTERVAL = 50

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

def extract_player_id(profile_url: str) -> str:
    """Extract the numeric player ID from a Transfermarkt profile URL."""
    match = re.search(r'/spieler/(\d+)', profile_url)
    return match.group(1) if match else ""


def scrape_positions(url: str, session: requests.Session) -> dict:
    """
    Scrape a single Transfermarkt player profile for position data.
    Returns dict with main_position and all_positions list.
    """
    result = {"main_position": "", "other_positions": []}

    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ERROR fetching {url}: {e}")
        return result

    soup = BeautifulSoup(resp.text, "html.parser")

    # --- Strategy 1: Look for the player info section with "Position:" label ---
    # Transfermarkt uses <li> items or <span> labels in the player data section.
    # The main position is typically labeled "Position:" and other positions as
    # "Other position:" or similar.

    # Try the info-table / player-data approach
    info_items = soup.select("li.data-header__label, span.data-header__label, div.data-header__label")
    for item in info_items:
        label_text = item.get_text(strip=True).lower()
        if "position" in label_text:
            # The value is usually in a sibling or child <span class="data-header__content">
            content = item.find_next("span", class_="data-header__content")
            if content:
                pos_text = content.get_text(strip=True)
                if "other" in label_text:
                    result["other_positions"].append(pos_text)
                else:
                    result["main_position"] = pos_text

    # --- Strategy 2: Look in the info table (older/alternate layout) ---
    if not result["main_position"]:
        for th in soup.find_all("th"):
            text = th.get_text(strip=True).lower()
            if "position" in text:
                td = th.find_next("td")
                if td:
                    pos_text = td.get_text(strip=True)
                    if "other" in text:
                        result["other_positions"].append(pos_text)
                    elif not result["main_position"]:
                        result["main_position"] = pos_text

    # --- Strategy 3: Look for the detail-position__box elements ---
    if not result["main_position"]:
        pos_boxes = soup.select(".detail-position__box, .detail-position__position")
        for box in pos_boxes:
            pos_text = box.get_text(strip=True)
            if pos_text:
                if not result["main_position"]:
                    result["main_position"] = pos_text
                else:
                    result["other_positions"].append(pos_text)

    # --- Strategy 4: Look for "Hauptposition" / "Nebenposition" (German labels) ---
    if not result["main_position"]:
        for element in soup.find_all(string=re.compile(r"(Hauptposition|Main position|Position)", re.I)):
            parent = element.parent
            if parent:
                sibling = parent.find_next_sibling()
                if sibling:
                    pos_text = sibling.get_text(strip=True)
                    if pos_text and not result["main_position"]:
                        result["main_position"] = pos_text

    # Look for other/secondary positions with German labels too
    for element in soup.find_all(string=re.compile(r"(Nebenposition|Other position)", re.I)):
        parent = element.parent
        if parent:
            sibling = parent.find_next_sibling()
            if sibling:
                pos_text = sibling.get_text(strip=True)
                if pos_text:
                    result["other_positions"].append(pos_text)

    # --- Strategy 5: Regex on raw HTML for position patterns ---
    if not result["main_position"]:
        # Look for patterns like "Position:</span>...<span>Right Winger</span>"
        html_text = resp.text
        main_match = re.search(
            r'(?:Position|Hauptposition)\s*:?\s*</(?:span|th|dt|div)>\s*<(?:span|td|dd|div)[^>]*>\s*([^<]+)',
            html_text, re.I
        )
        if main_match:
            result["main_position"] = main_match.group(1).strip()

        other_matches = re.findall(
            r'(?:Other position|Nebenposition)\s*:?\s*</(?:span|th|dt|div)>\s*<(?:span|td|dd|div)[^>]*>\s*([^<]+)',
            html_text, re.I
        )
        for m in other_matches:
            pos = m.strip()
            if pos:
                result["other_positions"].append(pos)

    return result


def load_checkpoint() -> dict:
    """Load checkpoint data if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"completed": {}, "last_index": -1}


def save_checkpoint(data: dict):
    """Save checkpoint data."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)


def save_results(results: list):
    """Save all results to CSV."""
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["player_id", "name", "main_position", "all_positions"])
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved {len(results)} rows to {OUTPUT_CSV}")


def main():
    # Read input CSV
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        players = list(reader)

    print(f"Loaded {len(players)} players from CSV")

    # Load checkpoint
    checkpoint = load_checkpoint()
    completed = checkpoint.get("completed", {})
    print(f"Found {len(completed)} already-scraped players in checkpoint")

    # Build results list from checkpoint
    results = []
    for pid, data in completed.items():
        results.append(data)

    # Create session for connection reuse
    session = requests.Session()

    # Process players
    new_count = 0
    for i, player in enumerate(players):
        profile_url = player.get("profile_url", "").strip()
        name = player.get("name", "").strip()
        csv_position = player.get("position", "").strip()

        if not profile_url:
            continue

        player_id = extract_player_id(profile_url)
        if not player_id:
            continue

        # Skip if already done
        if player_id in completed:
            continue

        print(f"[{i+1}/{len(players)}] Scraping: {name} (ID: {player_id})...")

        pos_data = scrape_positions(profile_url, session)

        # Use CSV position as fallback for main position
        main_pos = pos_data["main_position"] or csv_position
        other_pos = pos_data["other_positions"]

        # Build all_positions: main + others, deduplicated, preserving order
        all_positions = [main_pos] if main_pos else []
        for p in other_pos:
            if p and p not in all_positions:
                all_positions.append(p)

        row = {
            "player_id": player_id,
            "name": name,
            "main_position": main_pos,
            "all_positions": ", ".join(all_positions),
        }

        results.append(row)
        completed[player_id] = row
        new_count += 1

        # Checkpoint every N players
        if new_count % CHECKPOINT_INTERVAL == 0:
            print(f"  --- Checkpoint: {new_count} new players scraped, {len(results)} total ---")
            checkpoint["completed"] = completed
            checkpoint["last_index"] = i
            save_checkpoint(checkpoint)
            save_results(results)

        # Delay between requests
        time.sleep(DELAY)

    # Final save
    checkpoint["completed"] = completed
    save_checkpoint(checkpoint)
    save_results(results)

    print(f"\nDone! Scraped {new_count} new players. Total: {len(results)} rows.")
    print(f"Output: {OUTPUT_CSV}")

    # Clean up checkpoint file if all done
    if len(results) == len(players):
        print("All players scraped. You can delete the checkpoint file if you like:")
        print(f"  {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
