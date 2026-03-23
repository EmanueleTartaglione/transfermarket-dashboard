"""
Scrape Premier League player data from Transfermarkt.
Collects: name, position, age, nationality, club, market value, etc.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import json

BASE_URL = "https://www.transfermarkt.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


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


def get_team_links():
    """Get all Premier League team URLs from the league page."""
    url = f"{BASE_URL}/premier-league/startseite/wettbewerb/GB1"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    teams = []
    seen = set()

    # Find team links matching /startseite/verein/ pattern
    for link in soup.find_all("a", href=re.compile(r"/startseite/verein/\d+")):
        href = link.get("href", "")
        name = link.text.strip()
        if not name or len(name) < 2:
            name = link.get("title", "")
        if not name or len(name) < 2:
            continue

        # Normalize URL — remove saison_id params for consistency
        base_href = re.sub(r"/saison_id/\d+", "", href)
        team_id = re.search(r"/verein/(\d+)", base_href)
        if not team_id:
            continue
        tid = team_id.group(1)

        if tid not in seen:
            seen.add(tid)
            teams.append({"name": name, "url": BASE_URL + href})

    return teams


def get_team_players(team_name, team_url):
    """Scrape all players from a team's squad page."""
    # Convert startseite URL to kader (squad) URL for detailed view
    kader_url = team_url.replace("/startseite/", "/kader/")
    if "plus/1" not in kader_url:
        kader_url += "/plus/1"  # Detailed view

    print(f"  Scraping: {team_name} -> {kader_url}")
    resp = requests.get(kader_url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    players = []
    table = soup.find("table", class_="items")
    if not table:
        print(f"  WARNING: No player table found for {team_name}")
        return players

    tbody = table.find("tbody")
    if not tbody:
        return players

    rows = tbody.find_all("tr", class_=["odd", "even"])

    for row in rows:
        tds = row.find_all("td")
        if len(tds) < 13:
            continue

        player = {"club": team_name}

        # td[0]: shirt number
        player["shirt_number"] = tds[0].text.strip()

        # td[1] (posrela): contains inline-table with name + position
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

        # td[3] (hauptlink): player name (backup)
        if "name" not in player:
            name_link = tds[3].find("a")
            if name_link:
                player["name"] = name_link.text.strip()

        # td[4]: position (backup)
        if "position" not in player:
            player["position"] = tds[4].text.strip()

        # td[5]: date of birth + age, format "29/09/2000 (25)"
        dob_text = tds[5].text.strip()
        dob_match = re.match(r"(\d{2}/\d{2}/\d{4})\s*\((\d+)\)", dob_text)
        if dob_match:
            player["date_of_birth"] = dob_match.group(1)
            player["age"] = int(dob_match.group(2))

        # td[6]: nationality (flag images)
        flags = tds[6].find_all("img")
        nationalities = [f.get("title", "") for f in flags if f.get("title")]
        if nationalities:
            player["nationality"] = ", ".join(nationalities)

        # td[7]: height
        height_text = tds[7].text.strip()
        if height_text:
            player["height_m"] = height_text.replace(",", ".").replace("m", "").strip()

        # td[8]: preferred foot
        foot_text = tds[8].text.strip().lower()
        if foot_text in ["right", "left", "both"]:
            player["foot"] = foot_text

        # td[9]: joined date
        joined_text = tds[9].text.strip()
        if joined_text:
            player["joined"] = joined_text

        # td[10]: signed from (club logo)
        signed_from_img = tds[10].find("img")
        if signed_from_img:
            player["signed_from"] = signed_from_img.get("title", "")

        # td[11]: contract expires
        contract_text = tds[11].text.strip()
        if contract_text:
            player["contract_expires"] = contract_text

        # td[12]: market value
        mv_text = tds[12].text.strip()
        val = parse_market_value(mv_text)
        if val is not None:
            player["market_value_eur"] = val
            player["market_value_raw"] = mv_text

        if "name" in player:
            players.append(player)

    return players


def main():
    print("=" * 60)
    print("Transfermarkt Premier League Scraper")
    print("=" * 60)

    # Step 1: Get all team links
    print("\n[1/2] Fetching Premier League teams...")
    teams = get_team_links()
    print(f"Found {len(teams)} teams\n")

    if not teams:
        print("ERROR: No teams found. The page structure may have changed.")
        return

    for t in teams:
        print(f"  - {t['name']}")

    # Step 2: Scrape each team's players
    print(f"\n[2/2] Scraping player data from each team...")
    all_players = []

    for i, team in enumerate(teams, 1):
        print(f"\n[{i}/{len(teams)}] {team['name']}")
        try:
            players = get_team_players(team["name"], team["url"])
            all_players.extend(players)
            print(f"  Found {len(players)} players")
        except Exception as e:
            print(f"  ERROR scraping {team['name']}: {e}")

        # Be polite — wait between requests
        time.sleep(2)

    # Step 3: Save to CSV
    if all_players:
        df = pd.DataFrame(all_players)

        # Reorder columns
        preferred_cols = [
            "name", "club", "position", "age", "date_of_birth",
            "nationality", "height_m", "foot", "shirt_number",
            "market_value_eur", "market_value_raw",
            "joined", "signed_from", "contract_expires",
            "profile_url"
        ]
        cols = [c for c in preferred_cols if c in df.columns]
        extra = [c for c in df.columns if c not in preferred_cols]
        df = df[cols + extra]

        output_path = "premier_league_players.csv"
        df.to_csv(output_path, index=False)

        print("\n" + "=" * 60)
        print(f"SUCCESS! Scraped {len(all_players)} players from {len(teams)} teams")
        print(f"Saved to: {output_path}")
        print("=" * 60)

        # Quick summary
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nTop 10 most valuable players:")
        if "market_value_eur" in df.columns:
            top = df.nlargest(10, "market_value_eur")[["name", "club", "position", "age", "market_value_raw"]]
            print(top.to_string(index=False))
    else:
        print("\nNo player data collected. Something went wrong.")


if __name__ == "__main__":
    main()
