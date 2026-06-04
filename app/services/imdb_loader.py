#
# IMDB LOADER
# contcatenate with  MovieLens on links.csv (movielens_id @ imdb_id).


from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd
import ast

from app.schemas import ImdbRecord


def load_imdb_enrichment(
    imdb_parquet_path: Path,
    links_csv_path: Path,
) -> dict[int, ImdbRecord]:
    

    #Return {movielens_movie_id: ImdbRecord}
    
    if not imdb_parquet_path.exists():
        print(f"[imdb] No parquet at {imdb_parquet_path}; skipping enrichment.")
        return {}

    imdb_df = pd.read_parquet(imdb_parquet_path)
    links = pd.read_csv(links_csv_path, dtype={"imdbId": "Int64"})
    print(f"[imdb] Loaded parquet: {len(imdb_df)} rows")

    def normalize_imdb_id(v) -> Optional[int]:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        s = str(v).strip().lower()
        if s.startswith("tt"):
            s = s[2:]
            try:
                return int(s)
            except ValueError:
                return None
        try:
            return int(float(s))
        except ValueError:
            return None

    imdb_df["_imdb_int"] = imdb_df["imdbId"].map(normalize_imdb_id)
    imdb_df = imdb_df.dropna(subset=["_imdb_int"])
    imdb_df["_imdb_int"] = imdb_df["_imdb_int"].astype(int)

    imdb_lookup: dict[int, ImdbRecord] = {}
    
    for _, row in imdb_df.iterrows():
        # Nested Rating Parsing 
        raw_rating = row.get("imdbrating")
        parsed_rating = None
        parsed_votes = None

        if pd.notna(raw_rating):
            if isinstance(raw_rating, str):
                try:
                    raw_rating = ast.literal_eval(raw_rating)
                except (ValueError, SyntaxError):
                    raw_rating = {}
            
            if isinstance(raw_rating, dict):
                parsed_rating = raw_rating.get("rating")
                parsed_votes = raw_rating.get("numberofvotes")

        #Plot Concatenation Logic
        plot_parts = []
        raw_plot = row.get("plot")
        raw_plot_med = row.get("plotmedium")

        # Safely evaluate and append the standard plot
        if pd.notna(raw_plot) and str(raw_plot).strip() != "None":
            plot_parts.append(str(raw_plot).strip())
            
        # Safely evaluate the medium plot
        if pd.notna(raw_plot_med) and str(raw_plot_med).strip() != "None":
            clean_med = str(raw_plot_med).strip()
            # Only append if it actually adds new information (isn't a duplicate)
            if clean_med not in plot_parts:
                plot_parts.append(clean_med)

        # Join valid parts with a space. If empty, return None to satisfy API signature.
        final_plot = " ".join(plot_parts) if plot_parts else None

        #Record Instantiation
        rec = ImdbRecord(
            plot=final_plot,
            directors=[],
            cast=[],
            keywords=[],
            imdb_rating=float(parsed_rating) if parsed_rating is not None else None,
            num_votes=int(parsed_votes) if parsed_votes is not None else None,
            runtime_minutes=None,
        )
        
        imdb_lookup[int(row["_imdb_int"])] = rec

    # Join to MovieLens Data
    out: dict[int, ImdbRecord] = {}
    matched = 0
    
    # loop through your `links.csv` file. and map the cleaned `ImdbRecord` to movieId
    for _, row in links.iterrows():
        if pd.isna(row["imdbId"]):
            continue
            
        rec = imdb_lookup.get(int(row["imdbId"]))
        if rec is not None:
            out[int(row["movieId"])] = rec
            matched += 1

    print(f"[imdb] Matched {matched}/{len(links)} MovieLens movies to IMDB records")
    return out