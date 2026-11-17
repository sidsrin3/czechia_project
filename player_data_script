import os
import ssl
import numpy as np
import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
YEAR_START, YEAR_END = 2000, 2025   # inclusive: the ONLY line to change for a new window
TOP_N = 100                          # "elite" threshold for the top-100 flag

START, END = YEAR_START * 10000 + 101, YEAR_END * 10000 + 1231   # YYYYMMDD
MATCH_YEARS = range(YEAR_START, YEAR_END + 1)
LABEL = f"{YEAR_START}_{YEAR_END}"
BEST_COL  = f"best_rank_{LABEL}"
TOP_COL   = f"top100_{LABEL}"
WEEKS_COL = f"weeks_top100_{LABEL}"
RANKED_COL = f"weeks_ranked_{LABEL}"
RANK_ERAS = ['70s', '80s', '90s', '00s', '10s', '20s', 'current']  # missing files are skipped

# To use cloned folders instead of downloading, set ATP_DIR / WTA_DIR env vars
TOURS = {
    'ATP': {'prefix': 'atp',
            'url': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/',
            'local': os.environ.get('ATP_DIR')},
    'WTA': {'prefix': 'wta',
            'url': 'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/',
            'local': os.environ.get('WTA_DIR')},
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def read(tour, fname, names=None, must_have=()):
    """Read a CSV from the repo (or local folder). Handles encoding and missing headers."""
    cfg = TOURS[tour]
    path = f"{cfg['local']}/{fname}" if cfg['local'] else cfg['url'] + fname
    try:
        try:
            df = pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin-1', low_memory=False)
    except Exception as e:
        print(f"    could not load {fname}: {str(e)[:70]}")
        return None
    if names and must_have and not any(c in df.columns for c in must_have):
        try:
            df = pd.read_csv(path, header=None, names=names, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, header=None, names=names,
                             encoding='latin-1', low_memory=False)
    return df

def pick(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found. Columns are: {list(df.columns)}")

def to_int(s):
    return pd.to_numeric(s, errors='coerce')

# ─────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────
def load_rankings(tour):
    prefix = TOURS[tour]['prefix']
    frames = []
    for era in RANK_ERAS:
        df = read(tour, f"{prefix}_rankings_{era}.csv",
                  names=['ranking_date', 'rank', 'player', 'points'],
                  must_have=('ranking_date',))
        if df is None:
            continue
        out = pd.DataFrame({
            'date': to_int(df[pick(df, 'ranking_date')]),
            'rank': to_int(df[pick(df, 'rank', 'ranking')]),
            'pid':  to_int(df[pick(df, 'player', 'player_id')]),
        }).dropna().astype(int)
        print(f"    rankings_{era}: {len(out):,} rows "
              f"({out['date'].min()} to {out['date'].max()})")
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else None

def load_players(tour):
    prefix = TOURS[tour]['prefix']
    df = read(tour, f"{prefix}_players.csv",
              names=['player_id', 'name_first', 'name_last', 'hand', 'dob',
                     'ioc', 'height', 'wikidata_id'],
              must_have=('player_id',))
    first = pick(df, 'name_first', 'first_name')
    last = pick(df, 'name_last', 'last_name')
    ioc = pick(df, 'ioc', 'country_code')
    return pd.DataFrame({
        'pid': to_int(df[pick(df, 'player_id')]),
        'player_name': (df[first].fillna('').astype(str) + ' ' +
                        df[last].fillna('').astype(str)).str.strip(),
        'country_in_player_file': df[ioc],
    }).dropna(subset=['pid']).astype({'pid': int})

def load_match_countries(tour):
    """Country each player represented in matches during the window."""
    prefix = TOURS[tour]['prefix']
    pieces = []
    for yr in MATCH_YEARS:
        m = read(tour, f"{prefix}_matches_{yr}.csv")
        if m is None:
            continue
        try:
            w = pd.DataFrame({'pid': to_int(m['winner_id']), 'ioc': m['winner_ioc']})
            l = pd.DataFrame({'pid': to_int(m['loser_id']),  'ioc': m['loser_ioc']})
        except KeyError as e:
            print(f"    {prefix}_matches_{yr}.csv missing column {e}; skipping")
            continue
        pieces.append(pd.concat([w, l]))
    if not pieces:
        return None
    pc = pd.concat(pieces).dropna().astype({'pid': int})
    cnt = (pc.groupby(['pid', 'ioc']).size().reset_index(name='n')
             .sort_values(['pid', 'n'], ascending=[True, False]))
    top = cnt.drop_duplicates('pid')[['pid', 'ioc']].rename(columns={'ioc': 'country_matches'})
    ncty = cnt.groupby('pid')['ioc'].nunique().rename('n_countries').reset_index()
    return top.merge(ncty, on='pid')

# ─────────────────────────────────────────────
# BUILD ONE TOUR
# ─────────────────────────────────────────────
def build(tour):
    print(f"\n=== {tour} ===")
    print("  Loading rankings...")
    rk = load_rankings(tour)
    if rk is None:
        print("  No rankings loaded; skipping tour.")
        return None
    print(f"  Rankings span {rk['date'].min()} to {rk['date'].max()} "
          f"(career highs only reflect this span)")

    win = rk[(rk['date'] >= START) & (rk['date'] <= END)]
    if win.empty:
        print("  No ranking rows inside the window; skipping tour.")
        return None

    agg = win.groupby('pid').agg(
        best_rank_window=('rank', 'min'),
        weeks_ranked_window=('rank', 'size'),
    )
    agg['weeks_top100_window'] = (win.assign(t=win['rank'] <= TOP_N)
                                     .groupby('pid')['t'].sum().astype(int))
    agg = agg.reset_index()

    # All-time career high (within the files that loaded)
    ch = (rk.sort_values(['pid', 'rank', 'date']).drop_duplicates('pid')
            [['pid', 'rank', 'date']]
            .rename(columns={'rank': 'career_high_rank', 'date': 'career_high_date'}))

    print("  Loading players...")
    players = load_players(tour)
    print("  Loading match files for country at the time...")
    mc = load_match_countries(tour)

    df = agg.merge(ch, on='pid', how='left').merge(players, on='pid', how='left')
    if mc is not None:
        df = df.merge(mc, on='pid', how='left')
    else:
        df['country_matches'] = np.nan
        df['n_countries'] = np.nan

    df['country_ioc'] = df['country_matches'].fillna(df['country_in_player_file'])
    df['country_source'] = np.where(df['country_matches'].notna(),
                                    'matches in window', 'player file (current)')
    df['switched_country'] = df['n_countries'].fillna(1) > 1
    df['differs_from_player_file'] = (df['country_matches'].notna() &
                                      (df['country_matches'] != df['country_in_player_file']))
    df['top100_in_window'] = df['best_rank_window'] <= TOP_N
    df['tour'] = tour

    cols = ['tour', 'pid', 'player_name', 'country_ioc', 'country_source',
            'country_in_player_file', 'switched_country', 'differs_from_player_file',
            'best_rank_window', 'career_high_rank', 'career_high_date',
            'weeks_top100_window', 'weeks_ranked_window', 'top100_in_window']
    df = df[cols].rename(columns={
        'pid': 'player_id',
        'best_rank_window': BEST_COL,
        'weeks_top100_window': WEEKS_COL,
        'weeks_ranked_window': RANKED_COL,
        'top100_in_window': TOP_COL,
    }).sort_values(BEST_COL).reset_index(drop=True)

    df.to_csv(f"players_{LABEL}_{tour.lower()}.csv", index=False)
    print(f"  Saved: players_{LABEL}_{tour.lower()}.csv")
    return df

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
results = [r for r in (build(t) for t in TOURS) if r is not None]
if not results:
    raise SystemExit("\nNothing built. Check the messages above.")

combined = pd.concat(results, ignore_index=True)
combined.to_csv(f"players_{LABEL}_combined.csv", index=False)

print("\n─── SUMMARY ───")
for tour, g in combined.groupby('tour'):
    print(f"{tour}: {len(g):,} ranked players in {YEAR_START}-{YEAR_END}, "
          f"{int(g[TOP_COL].sum())} reached the top {TOP_N}")

top = combined[combined[TOP_COL]]
print(f"\nTop-{TOP_N} players by country and tour (distinct players):")
tbl = (top.groupby(['country_ioc', 'tour']).size().unstack(fill_value=0))
tbl['total'] = tbl.sum(axis=1)
print(tbl.sort_values('total', ascending=False).head(15).to_string())

legacy = {'YUG', 'SCG', 'SRB', 'MNE', 'TCH', 'URS', 'FRG', 'GDR'}
leg = top[top['country_ioc'].isin(legacy)]
if len(leg):
    print("\nCodes needing a manual decision (state changes around this era):")
    print(leg.groupby(['country_ioc', 'tour']).size().to_string())

print(f"\nPlayers who changed country in the window: "
      f"{int(combined['switched_country'].sum())}")
print(f"Players whose country differs from the player file: "
      f"{int(combined['differs_from_player_file'].sum())}")
print(f"\nSaved: players_{LABEL}_combined.csv")
