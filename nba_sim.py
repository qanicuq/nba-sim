import json
import os
import sys
import time
import urllib.error
import urllib.request
from bisect import bisect_left
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates="
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
GAMES_PATH = "data/games.csv"
KALSHI_PATH = "data/kalshi_games.csv.gz"
SERIES = "KXNBAGAME"
N_SIMS = 10000
ET = ZoneInfo("America/New_York")
# English month names so the event key matches Kalshi (NOV) on any locale.
MONTHS = (
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
)
NBA_TEAMS = (
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NO", "NY",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SA", "TOR", "UTAH", "WSH",
)
KALSHI_TEAM = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
}
ESPN_FROM_KALSHI = {kalshi: espn for espn, kalshi in KALSHI_TEAM.items()}
KALSHI_COLUMNS = [
    "event_ticker", "ticker", "team", "game_id", "tipoff_utc", "result", "yes_ask",
]
GAME_COLUMNS = [
    "game_id", "season", "tipoff_utc", "home_team", "away_team",
    "home_score", "away_score", "neutral",
]


def get_json(url):
    for attempt in range(6):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "nba-sim"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError("too many retries")


def num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if 0 < out < 1 else None


def _event_code(away, home, when):
    date = f"{when.year % 100:02d}{MONTHS[when.month - 1]}{when.day:02d}"
    return date + KALSHI_TEAM.get(away, away) + KALSHI_TEAM.get(home, home)


def fetch_day(day, pre_only=False):
    payload = get_json(ESPN + day.strftime("%Y%m%d"))
    rows = []
    for event in payload.get("events") or []:
        season = event.get("season") or {}
        if season.get("type") == 1:
            continue
        status_type = (event.get("status") or {}).get("type") or {}
        if pre_only and status_type.get("state") != "pre":
            continue
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        home = None
        away = None
        for competitor in comp.get("competitors") or []:
            if competitor.get("homeAway") == "home":
                home = competitor
            elif competitor.get("homeAway") == "away":
                away = competitor
        if home is None or away is None:
            continue
        home_team = (home.get("team") or {}).get("abbreviation")
        away_team = (away.get("team") or {}).get("abbreviation")
        if home_team not in NBA_TEAMS or away_team not in NBA_TEAMS:
            continue
        completed = status_type.get("completed") is True
        if completed:
            home_score = int(home.get("score"))
            away_score = int(away.get("score"))
        else:
            home_score = None
            away_score = None
        rows.append({
            "game_id": str(event["id"]),
            "season": int(season["year"]),
            "tipoff_utc": event["date"],
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "neutral": bool(comp.get("neutralSite")),
        })
    return rows


def download():
    # Season year is the year the season ends. From July on, that is next year.
    today = datetime.now(ET).date()
    current = today.year + 1 if today.month >= 7 else today.year
    days = []
    for season_year in range(2016, current + 1):
        start = datetime(season_year - 1, 10, 1).date()
        end = min(datetime(season_year, 6, 30).date(), today)
        day = start
        while day <= end:
            days.append(day)
            day += timedelta(days=1)
    old = None
    if os.path.exists(GAMES_PATH):
        old = pd.read_csv(GAMES_PATH, dtype={"game_id": str})
        if len(old) and old["tipoff_utc"].notna().any():
            tips = pd.to_datetime(old["tipoff_utc"], utc=True).dt.tz_convert(ET)
            newest = tips.max().date()
            cutoff = newest - timedelta(days=2)
            days = [day for day in days if day >= cutoff]
    rows = []
    for index, day in enumerate(days):
        if index:
            time.sleep(0.2)
        rows.extend(fetch_day(day))
    new = pd.DataFrame(rows, columns=GAME_COLUMNS)
    if old is not None:
        frame = pd.concat([old, new], ignore_index=True)
    else:
        frame = new
    frame["game_id"] = frame["game_id"].astype(str)
    frame = frame.drop_duplicates("game_id", keep="last")
    frame = frame.sort_values("tipoff_utc").reset_index(drop=True)
    os.makedirs("data", exist_ok=True)
    frame.to_csv(GAMES_PATH, index=False)
    print(f"saved {len(frame)} games")


def load_games():
    games = pd.read_csv(GAMES_PATH, dtype={"game_id": str})
    tips = pd.to_datetime(games["tipoff_utc"], utc=True).dt.tz_convert(ET)
    games["gameday"] = pd.to_datetime(tips.dt.strftime("%Y-%m-%d"))
    neutral = games["neutral"]
    if neutral.dtype != bool:
        neutral = neutral.astype(str).str.strip().str.lower().isin(["true", "1"])
    games["location"] = np.where(neutral, "Neutral", "Home")
    games["season"] = games["season"].astype(int)
    games = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return games


def team_ratings_before(games, team, date, N):
    # Completed games only. Nothing on or after `date` is used.
    cached = getattr(team_ratings_before, "_hist", None)
    cached_id = getattr(team_ratings_before, "_hist_id", None)
    if cached is None or cached_id != id(games):
        hist = {}
        for row in games.itertuples(index=False):
            if pd.isna(row.home_score) or pd.isna(row.away_score):
                continue
            hist.setdefault(row.home_team, []).append(
                (row.gameday, float(row.home_score), float(row.away_score))
            )
            hist.setdefault(row.away_team, []).append(
                (row.gameday, float(row.away_score), float(row.home_score))
            )
        team_ratings_before._hist = hist
        team_ratings_before._hist_id = id(games)
    else:
        hist = cached

    history = hist.get(team)
    if not history:
        return None, None
    cut = bisect_left(history, (date,))
    prior = history[max(0, cut - N) : cut]
    if not prior:
        return None, None
    scored = sum(item[1] for item in prior) / len(prior)
    allowed = sum(item[2] for item in prior) / len(prior)
    return scored, allowed


def expected_points(games, game_row, N, H):
    if game_row["location"] == "Neutral":
        H = 0
    home_scored, home_allowed = team_ratings_before(
        games, game_row["home_team"], game_row["gameday"], N
    )
    away_scored, away_allowed = team_ratings_before(
        games, game_row["away_team"], game_row["gameday"], N
    )
    if home_scored is None or away_scored is None:
        return None, None
    home_exp = (home_scored + away_allowed) / 2 + H / 2
    away_exp = (away_scored + home_allowed) / 2 - H / 2
    return home_exp, away_exp


def sim_scores(home_exp, away_exp, S, rng):
    home_scores = np.maximum(np.rint(rng.normal(home_exp, S, N_SIMS)), 0)
    away_scores = np.maximum(np.rint(rng.normal(away_exp, S, N_SIMS)), 0)
    return home_scores, away_scores


def win_prob(margin):
    # Both sites settle a tie at 50 cents.
    return float(np.mean(margin > 0) + 0.5 * np.mean(margin == 0))


def kalshi_fee(price, multiplier):
    # Kalshi quadratic taker fee per contract (Kalshi rounds the order total up to a cent).
    return 0.07 * multiplier * price * (1 - price)


def _close_dollars(book):
    if not isinstance(book, dict):
        return None
    raw = book.get("close_dollars")
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _fee_multiplier():
    # edges.py reads every Sports series, then uses that series or 1.
    series_info = get_json(f"{KALSHI}/series?category=Sports")
    fee_mult = {
        item["ticker"]: float(item.get("fee_multiplier") or 1)
        for item in series_info.get("series", [])
    }
    return fee_mult.get(SERIES, 1.0)


def _settled_event_tickers(series):
    cursor = ""
    while True:
        url = f"{KALSHI}/events?series_ticker={series}&status=settled&limit=200"
        if cursor:
            url += "&cursor=" + cursor
        payload = get_json(url)
        time.sleep(0.3)
        for event in payload.get("events") or []:
            ticker = event.get("event_ticker")
            if ticker:
                yield ticker
        cursor = payload.get("cursor") or ""
        if not cursor:
            break


def kalshi_download():
    games = load_games()
    lookup = {}
    for row in games.itertuples(index=False):
        lookup[_event_code(row.away_team, row.home_team, row.gameday)] = row
    rows = []
    unmatched = []
    seen = 0
    for event_ticker in _settled_event_tickers(SERIES):
        seen += 1
        code = event_ticker.split("-", 1)[1] if "-" in event_ticker else ""
        found = lookup.get(code)
        if found is None:
            unmatched.append(event_ticker)
        else:
            tipoff = datetime.fromisoformat(str(found.tipoff_utc).replace("Z", "+00:00"))
            entry = int(tipoff.timestamp()) - 3600
            payload = get_json(f"{KALSHI}/markets?event_ticker={event_ticker}&limit=1000")
            time.sleep(0.3)
            markets = payload.get("markets") or []
            info = {}
            tickers = []
            for market in markets:
                ticker = market.get("ticker")
                if not ticker:
                    continue
                info[ticker] = market
                tickers.append(ticker)
            for start in range(0, len(tickers), 100):
                group = tickers[start:start + 100]
                url = (
                    f"{KALSHI}/markets/candlesticks?market_tickers={','.join(group)}"
                    f"&start_ts={entry - 6 * 3600}&end_ts={entry}&period_interval=60"
                )
                candles = get_json(url)
                time.sleep(0.3)
                for item in candles.get("markets") or []:
                    ticker = item.get("market_ticker")
                    market = info.get(ticker)
                    if market is None:
                        continue
                    chosen = None
                    for candle in item.get("candlesticks") or []:
                        ts = candle.get("end_period_ts")
                        if ts is None or ts > entry:
                            continue
                        if chosen is None or ts >= chosen["end_period_ts"]:
                            chosen = candle
                    if chosen is None:
                        continue
                    yes_ask = _close_dollars(chosen.get("yes_ask"))
                    if yes_ask is None:
                        continue
                    suffix = ticker.rsplit("-", 1)[-1]
                    team = ESPN_FROM_KALSHI.get(suffix, suffix)
                    if team != found.home_team and team != found.away_team:
                        continue
                    rows.append({
                        "event_ticker": event_ticker,
                        "ticker": ticker,
                        "team": team,
                        "game_id": str(found.game_id),
                        "tipoff_utc": found.tipoff_utc,
                        "result": market.get("result"),
                        "yes_ask": yes_ask,
                    })
        if seen % 50 == 0:
            print(f"{seen} events, {len(rows)} rows")
    os.makedirs("data", exist_ok=True)
    frame = pd.DataFrame(rows, columns=KALSHI_COLUMNS)
    frame = frame.drop_duplicates("ticker", keep="last")
    frame.to_csv(KALSHI_PATH, index=False, compression="gzip")
    print(f"saved {len(frame)} rows")
    print(f"unmatched events: {len(unmatched)}")
    for ticker in unmatched[:20]:
        print(ticker)


def _blank_month():
    return {
        "games": 0,
        "model_bets": 0,
        "model_wins": 0,
        "model_profit": 0.0,
        "base1": 0.0,
        "base2": 0.0,
    }


def backtest():
    games = load_games()
    played = games["home_score"].notna() & games["away_score"].notna()
    train = games[games["season"].between(2016, 2022) & played]
    train_scores = np.concatenate([
        train["home_score"].to_numpy(dtype=float),
        train["away_score"].to_numpy(dtype=float),
    ])
    S = float(np.std(train_scores, ddof=1))

    train_rows = train[
        ["home_team", "away_team", "gameday", "location", "home_score", "away_score"]
    ].to_dict("records")
    best_mae = None
    best_n = None
    best_h = None
    for N in (5, 10, 20, 40):
        for H in (0, 1, 2, 3, 4):
            errors = []
            for game_row in train_rows:
                home_exp, away_exp = expected_points(games, game_row, N, H)
                if home_exp is None:
                    continue
                actual = float(game_row["home_score"]) - float(game_row["away_score"])
                errors.append(abs((home_exp - away_exp) - actual))
            mae = float(np.mean(errors))
            if best_mae is None or mae < best_mae:
                best_mae = mae
                best_n = N
                best_h = H

    with open("settings.json", "w", encoding="utf-8") as handle:
        json.dump({"N": best_n, "H": best_h, "S": S}, handle)
        handle.write("\n")

    later = games[(games["season"] >= 2023) & played]
    later_rows = later[
        ["home_team", "away_team", "gameday", "location", "home_score", "away_score"]
    ].to_dict("records")
    model_errors = []
    base_errors = []
    for game_row in later_rows:
        home_exp, away_exp = expected_points(games, game_row, best_n, best_h)
        if home_exp is None:
            continue
        actual = float(game_row["home_score"]) - float(game_row["away_score"])
        model_errors.append(abs((home_exp - away_exp) - actual))
        base_errors.append(abs(float(best_h) - actual))
    model_mae = float(np.mean(model_errors))
    base_mae = float(np.mean(base_errors))

    kalshi = pd.read_csv(KALSHI_PATH, dtype={"game_id": str})
    mult = _fee_multiplier() if len(kalshi) else 1.0
    rng = np.random.default_rng(42)
    by_id = {str(row.game_id): row for row in games.itertuples(index=False)}
    months = {}
    for game_id, markets in kalshi.groupby("game_id", sort=False):
        game = by_id.get(str(game_id))
        if game is None:
            continue
        game_row = {
            "home_team": game.home_team,
            "away_team": game.away_team,
            "gameday": game.gameday,
            "location": game.location,
        }
        home_exp, away_exp = expected_points(games, game_row, best_n, best_h)
        if home_exp is None:
            continue
        home_scores, away_scores = sim_scores(home_exp, away_exp, S, rng)
        chance = {
            game.home_team: win_prob(home_scores - away_scores),
            game.away_team: win_prob(away_scores - home_scores),
        }
        month = f"{game.gameday.year:04d}-{game.gameday.month:02d}"
        if month not in months:
            months[month] = _blank_month()
        bucket = months[month]
        bucket["games"] += 1
        priced = []
        for market in markets.itertuples(index=False):
            yes_ask = float(market.yes_ask)
            fee = kalshi_fee(yes_ask, mult)
            team = market.team
            if team not in chance:
                continue
            edge = chance[team] - yes_ask - fee
            if market.result == "yes":
                profit = 1 - yes_ask - fee
            elif market.result == "no":
                profit = -yes_ask - fee
            else:
                profit = None
            priced.append({
                "team": team,
                "yes_ask": yes_ask,
                "edge": edge,
                "profit": profit,
                "result": market.result,
            })
        usable = [item for item in priced if item["profit"] is not None]
        if usable:
            best = max(usable, key=lambda item: item["edge"])
            if best["edge"] >= 0.05:
                bucket["model_bets"] += 1
                bucket["model_profit"] += best["profit"]
                if best["result"] == "yes":
                    bucket["model_wins"] += 1
        home_bet = next((item for item in priced if item["team"] == game.home_team), None)
        if home_bet is not None and home_bet["profit"] is not None:
            bucket["base1"] += home_bet["profit"]
        if priced:
            favorite = max(priced, key=lambda item: item["yes_ask"])
            if favorite["profit"] is not None:
                bucket["base2"] += favorite["profit"]

    totals = _blank_month()
    for bucket in months.values():
        for key in totals:
            totals[key] += bucket[key]

    def row_line(label, bucket):
        return (
            f"| {label} | {bucket['games']} | {bucket['model_bets']} | "
            f"{bucket['model_wins']} | {bucket['model_profit']:.3f} | "
            f"{bucket['base1']:.3f} | {bucket['base2']:.3f} |"
        )

    lines = [
        "# Backtest results",
        "",
        "N and H were chosen on 2016-2022 seasons only. Every Kalshi game here is newer, so none of it was used to choose the rules.",
        "",
        f"Chosen N: {best_n}",
        f"Chosen H: {best_h}",
        f"S: {S:.4f}",
        "",
        f"Mean absolute error of the model margin on seasons 2023 and later: {model_mae:.3f}",
        f"Mean absolute error of the baseline (home team wins by H) on seasons 2023 and later: {base_mae:.3f}",
        "",
        "## Month by month",
        "",
        "| year-month | games | model bets | model wins | model profit | baseline 1 profit | baseline 2 profit |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for month in sorted(months):
        lines.append(row_line(month, months[month]))
    lines.extend([
        "",
        "## Full period",
        "",
        "| year-month | games | model bets | model wins | model profit | baseline 1 profit | baseline 2 profit |",
        "|---|---:|---:|---:|---:|---:|---:|",
        row_line("full period", totals),
        "",
    ])
    with open("backtest_results.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _send_push(message):
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        print("push failed")
        return
    url = "https://ntfy.sh/" + topic
    request = urllib.request.Request(
        url, data=message.encode("utf-8"), method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
    except Exception:
        print("push failed")


def _clock(tip):
    hour = tip.hour % 12 or 12
    ampm = "AM" if tip.hour < 12 else "PM"
    return f"{tip.year:04d}-{tip.month:02d}-{tip.day:02d} {hour}:{tip.minute:02d} {ampm}"


def _cell(value, digits):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def predict():
    with open("settings.json", encoding="utf-8") as handle:
        settings = json.load(handle)
    N = int(settings["N"])
    H = settings["H"]
    S = float(settings["S"])

    today = datetime.now(ET).date()
    rows = fetch_day(today, pre_only=True)
    time.sleep(0.2)
    rows.extend(fetch_day(today + timedelta(days=1), pre_only=True))
    rows.sort(key=lambda row: row["tipoff_utc"])
    games = load_games()
    rng = np.random.default_rng(42)
    ready = []
    for row in rows:
        tip = datetime.fromisoformat(row["tipoff_utc"].replace("Z", "+00:00")).astimezone(ET)
        game_row = {
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "gameday": pd.Timestamp(tip.date()),
            "location": "Neutral" if row["neutral"] else "Home",
        }
        home_exp, away_exp = expected_points(games, game_row, N, H)
        if home_exp is None:
            continue
        home_scores, away_scores = sim_scores(home_exp, away_exp, S, rng)
        ready.append({
            "row": row,
            "tip": tip,
            "away_chance": win_prob(away_scores - home_scores),
            "home_chance": win_prob(home_scores - away_scores),
            "away_mean": float(away_scores.mean()),
            "home_mean": float(home_scores.mean()),
        })

    mult = _fee_multiplier() if ready else 1.0
    output_rows = []
    alert_lines = []
    for index, item in enumerate(ready):
        row = item["row"]
        tip = item["tip"]
        if index:
            time.sleep(0.3)
        event_ticker = "KXNBAGAME-" + _event_code(row["away_team"], row["home_team"], tip)
        payload = get_json(f"{KALSHI}/markets?event_ticker={event_ticker}&limit=1000")
        markets = payload.get("markets") or []
        asks = {}
        edges = {}
        for market in markets:
            ticker = market.get("ticker") or ""
            suffix = ticker.rsplit("-", 1)[-1]
            team = ESPN_FROM_KALSHI.get(suffix, suffix)
            if team != row["away_team"] and team != row["home_team"]:
                continue
            yes_ask = num(market.get("yes_ask_dollars"))
            if yes_ask is None:
                continue
            sim = item["away_chance"] if team == row["away_team"] else item["home_chance"]
            asks[team] = yes_ask
            edges[team] = sim - yes_ask - kalshi_fee(yes_ask, mult)
        game = f"{row['away_team']} @ {row['home_team']} {_clock(tip)}"
        if not markets:
            pick = "no Kalshi market"
            best_pts = None
        elif not edges:
            pick = "no bet"
            best_pts = None
        else:
            team = max(edges, key=edges.get)
            best_pts = 100.0 * edges[team]
            pick = team if edges[team] >= 0.05 else "no bet"
        output_rows.append({
            "game": game,
            "away_pct": 100.0 * item["away_chance"],
            "home_pct": 100.0 * item["home_chance"],
            "score": f"{item['away_mean']:.1f}-{item['home_mean']:.1f}",
            "away_ask": asks.get(row["away_team"]),
            "home_ask": asks.get(row["home_team"]),
            "best_pts": best_pts,
            "pick": pick,
        })
        if pick != "no bet" and pick != "no Kalshi market":
            alert_lines.append(f"{game}: {pick}")

    lines = [
        "# Predictions",
        "",
        "Prices are the Kalshi YES ask at run time, fees taken out.",
        "This model has not proven it beats Kalshi. Check backtest_results.md before betting.",
        "",
        "| game | sim away win % | sim home win % | sim average score away-home | Kalshi away YES ask | Kalshi home YES ask | best edge in points | pick |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in output_rows:
        lines.append(
            "| {game} | {away} | {home} | {score} | {away_ask} | {home_ask} | {edge} | {pick} |".format(
                game=row["game"],
                away=_cell(row["away_pct"], 1),
                home=_cell(row["home_pct"], 1),
                score=row["score"],
                away_ask=_cell(row["away_ask"], 2),
                home_ask=_cell(row["home_ask"], 2),
                edge=_cell(row["best_pts"], 1),
                pick=row["pick"],
            )
        )
    lines.append("")
    with open("predictions.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    if alert_lines:
        _send_push("\n".join(alert_lines))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "download":
        download()
    elif len(sys.argv) == 2 and sys.argv[1] == "kalshi_download":
        kalshi_download()
    elif len(sys.argv) == 2 and sys.argv[1] == "backtest":
        backtest()
    elif len(sys.argv) == 2 and sys.argv[1] == "predict":
        predict()
    else:
        sys.exit(1)
