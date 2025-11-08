#!/usr/bin/env python

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

LASTFM_ENDPOINT = "https://ws.audioscrobbler.com/2.0/"
LASTFM_API_KEY = None  # will be filled in main()


# ---------- CLI ARGUMENTS ----------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Enrich dataset.csv with Last.fm metadata in batch mode."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset.csv",
        help="Path to input CSV (original Spotify dataset).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset_with_lastfm.csv",
        help="Path to output CSV (will be created/updated).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=-1,
        help="Number of tracks to process (-1 = all remaining).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save checkpoint to --output after this many processed tracks.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Seconds to sleep between API calls (rate limiting).",
    )
    parser.add_argument(
        "--api-env",
        type=str,
        default="api.env",
        help="Path to api.env file (used if LASTFM_API_KEY env var is not set).",
    )
    return parser.parse_args()


# ---------- API KEY LOADING ----------

def get_lastfm_api_key(api_env_path: str) -> str:
    """
    1) Try environment variable LASTFM_API_KEY.
    2) Fallback to api.env file with line: LASTFM_API_KEY=...
    """
    key = os.getenv("LASTFM_API_KEY")
    if key:
        return key

    env_path = Path(api_env_path)
    if env_path.exists():
        env_vars = {}
        with env_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()
        key = env_vars.get("LASTFM_API_KEY")
        if key:
            return key

    raise RuntimeError(
        "LASTFM_API_KEY not found. Either export it in the environment or put it "
        f"in {api_env_path} as LASTFM_API_KEY=your_key_here"
    )


# ---------- NORMALIZATION / TAG HELPERS ----------

def normalize_artist_name(raw: str) -> str:
    """
    Take the Spotify 'artists' field and turn it into something Last.fm likes.
    We just keep the primary artist and strip 'feat.' / 'ft.' bits.
    """
    if not isinstance(raw, str):
        return str(raw)

    # Take the first artist if there are multiple separated by ; or ,
    primary = raw.split(";")[0].split(",")[0].strip()

    # Strip common 'feat.' patterns
    lowered = primary.lower()
    for kw in [" feat. ", " feat ", " ft. ", " ft "]:
        if kw.strip() in lowered:
            idx = lowered.find(kw.strip())
            primary = primary[:idx].strip()
            break

    return primary


def get_artist_top_tags(artist_name: str, max_tags: int = 10):
    params = {
        "method": "artist.getTopTags",
        "api_key": LASTFM_API_KEY,
        "artist": artist_name,
        "autocorrect": 1,
        "format": "json",
    }
    try:
        r = requests.get(LASTFM_ENDPOINT, params=params, timeout=5)
        r.raise_for_status()
        payload = r.json()

        if "error" in payload:
            return []

        raw_tags = payload.get("toptags", {}).get("tag", [])
        if isinstance(raw_tags, dict):
            raw_tags = [raw_tags]

        tags = []
        for t in raw_tags[:max_tags]:
            name = t.get("name")
            if isinstance(name, str):
                tags.append(name.lower())
        return tags
    except Exception as e:
        print("Artist tags failed:", e, "for artist:", artist_name)
        return []


# ---------- LAST.FM TRACK METADATA ----------

def get_lastfm_metadata(artist_name: str, song_title: str):
    params = {
        "method": "track.getInfo",
        "api_key": LASTFM_API_KEY,
        "artist": artist_name,
        "track": song_title,
        "autocorrect": 1,
        "format": "json",
    }

    try:
        response = requests.get(LASTFM_ENDPOINT, params=params, timeout=5)
        response.raise_for_status()
        payload = response.json()

        if "error" in payload:
            print("Last.fm error for", artist_name, "-", song_title, ":", payload)
            return None

        track_obj = payload.get("track", {})

        playcount = track_obj.get("playcount")
        listeners = track_obj.get("listeners")
        duration = track_obj.get("duration")

        playcount = int(playcount) if playcount is not None else None
        listeners = int(listeners) if listeners is not None else None
        duration = int(duration) if duration is not None else None

        # 1) Track-level tags
        raw_tags = track_obj.get("toptags", {}).get("tag", [])
        if isinstance(raw_tags, dict):
            raw_tags = [raw_tags]

        tag_names = []
        for t in raw_tags:
            name = t.get("name")
            if isinstance(name, str):
                tag_names.append(name.lower())

        # 2) Fallback: artist-level tags if track tags missing
        if not tag_names:
            artist_tags = get_artist_top_tags(artist_name)
            tag_names.extend(artist_tags)

        tag_string = ";".join(tag_names) if tag_names else None

        return {
            "lfm_playcount": playcount,
            "lfm_listeners": listeners,
            "lfm_duration_ms": duration,  # Last.fm duration is usually in ms
            "lfm_tags": tag_string,
        }

    except Exception as e:
        print("Request failed for", artist_name, "-", song_title, ":", e)
        return None


# ---------- DATAFRAME LOADING / RESUME LOGIC ----------

LASTFM_COLS = ["lfm_playcount", "lfm_listeners", "lfm_duration_ms", "lfm_tags"]


def load_or_init_dataframe(input_path: str, output_path: str) -> pd.DataFrame:
    """
    If output_path exists, resume from it.
    Otherwise, load input_path and add empty Last.fm columns.
    No cleaning is done here: we just use dataset.csv directly.
    """
    out_path = Path(output_path)
    if out_path.exists():
        print(f"[INFO] Resuming from existing {output_path}")
        df = pd.read_csv(out_path)
    else:
        print(f"[INFO] Loading fresh input from {input_path}")
        df = pd.read_csv(input_path)
        # ensure Last.fm columns exist
        for col in LASTFM_COLS:
            if col not in df.columns:
                df[col] = pd.NA

    return df


# ---------- MAIN LOOP ----------

def main():
    global LASTFM_API_KEY

    args = parse_args()
    LASTFM_API_KEY = get_lastfm_api_key(args.api_env)

    df = load_or_init_dataframe(args.input, args.output)

    # Choose rows that still need Last.fm data (all 4 cols NaN/NA)
    mask_unprocessed = (
        df["lfm_playcount"].isna()
        & df["lfm_listeners"].isna()
        & df["lfm_duration_ms"].isna()
        & df["lfm_tags"].isna()
    )

    candidate_indices = df.index[mask_unprocessed].tolist()

    if args.n > 0:
        candidate_indices = candidate_indices[: args.n]

    total_to_do = len(candidate_indices)
    print(f"[INFO] Tracks to process in this run: {total_to_do}")

    processed = 0
    start_time = time.time()

    for i, idx in enumerate(candidate_indices, start=1):
        row = df.loc[idx]

        song_title = row["track_name"]
        artist_raw = row["artists"]
        main_artist = normalize_artist_name(artist_raw)

        meta = get_lastfm_metadata(main_artist, song_title)

        if meta is not None:
            df.at[idx, "lfm_playcount"] = meta["lfm_playcount"]
            df.at[idx, "lfm_listeners"] = meta["lfm_listeners"]
            df.at[idx, "lfm_duration_ms"] = meta["lfm_duration_ms"]
            df.at[idx, "lfm_tags"] = meta["lfm_tags"]

        processed += 1

        # Simple progress log to stdout (shows up in Slurm output)
        if processed % 10 == 0 or processed == total_to_do:
            elapsed = time.time() - start_time
            print(
                f"[INFO] Processed {processed}/{total_to_do} "
                f"tracks (elapsed {elapsed:.1f} s)"
            )

        # Periodic checkpoint
        if processed % args.save_every == 0:
            df.to_csv(args.output, index=False)
            print(
                f"[INFO] Saved checkpoint to {args.output} "
                f"after {processed} tracks"
            )

        # Be polite to the API
        time.sleep(args.sleep)

    # Final save
    df.to_csv(args.output, index=False)
    print(
        f"[INFO] Done. Saved {len(df)} rows and {len(df.columns)} columns to {args.output}"
    )


if __name__ == "__main__":
    main()
