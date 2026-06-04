#TASK 3 — Task 1 with IMDB-enriched content-based movie

#Run: python -m scripts.build_task3

## This is the same file as build_task1.py with minor changes

import re
import sys
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.vector_store import get_store
from app.services.imdb_loader import load_imdb_enrichment

from app.schemas import ImdbRecord

# defined here as global to be compiled once to avoid re-parsing on every movie
YEAR_REGEX = re.compile(r"\((\d{4})\)\s*$")


def parse_title_year(raw: str) -> tuple[str, int | None]:
    # preprocess the title and year and split them for each movie. 
    # this assume that all movies in the dataset is of the form  "movie (year)". will fail if the () are missing
    # should try catch if errors occurs in production
    match = YEAR_REGEX.search(raw.strip())
    if match:
        return YEAR_REGEX.sub("", raw).strip(), int(match.group(1))
    return raw.strip(), None


def parse_genres(raw: str) -> list[str]:
    # we assume after a quick examination of the dataset that all genra are of the for  A|B|C or empty
    # should try catch if errors occurs in production
    if not raw or raw == "(no genres listed)":
        return []
    return [g.strip().lower() for g in raw.split("|") if g.strip()]


def build_description(
    title: str,
    year: int | None,
    genres: list[str],
    top_tags: list[str],
    avg_rating: float | None,
    num_ratings: int,
    imdb: ImdbRecord | None,
) -> str:

    # NOTE
    # here we will createa vectors embedding for searching.
    # to be able to work with accurate vectors we need to append our data we extracted into well formed english sentence.
    # for this i will use a generate sentence template. BUT to get better results you can use llms to generate a factual well sentences that are more descriptive.
    # This is needed because embedding vectors are trained on actual english sentences and not words.
    # ORDERING: plot first, then cast/directors, then MovieLens fields,stats last.   plot most important so it is 1st.
    # this is very basic but can be enhanced with LLMS
    # NOTE if if a movie has no IMDB match then only use MovieLens data like in TASk1

    parts = [f"{title} ({year})." if year else f"{title}."]

    # IMDB plot 
    if imdb and imdb.plot:
        plot = imdb.plot.strip()
        # Truncate to keep descriptions in the embedding model's sweet spot (~256 tokens)
        # NOTE this is dependent of the model my pc can't handle a larger model. on a production machine you should test both  small and large report performance then pick the best result with respect to cost.
        if len(plot) > 600:
            plot = plot[:600].rsplit(".", 1)[0] + "."
        parts.append(plot)

    # Directors this will allow to have the film for the same director close together.
    if imdb and imdb.directors:
        parts.append(f"Directed by {', '.join(imdb.directors[:2])}.")

    # Cast pick only top 3 only or just the most prominent. if we have a larger model we can erform testing and larger vectors
    if imdb and imdb.cast:
        parts.append(f"Starring {', '.join(imdb.cast[:3])}.")

    if genres:
        parts.append(
            f"A {genres[0]} film." if len(genres) == 1
            else f"Genres: {', '.join(genres)}."
        )

    # Merge IMDB keywords + MovieLens tags, dedup case-insensitively, cap at 8
    combined, seen = [], set()
    for t in (imdb.keywords if imdb else []) + top_tags:
        t = t.lower().strip()
        if t and t not in seen:
            combined.append(t)
            seen.add(t)
        if len(combined) >= 8:
            break
    if combined:
        parts.append(f"Themes and keywords: {', '.join(combined)}.")

    if avg_rating is not None and num_ratings >= 5:
        parts.append(
            f"MovieLens users rate it {avg_rating:.1f} out of 5 across {num_ratings} ratings."
        )

    # IMDB quality signal sentence wise as explained earlier  example :iMDB score: 8.4 from 12,000 voters.
    if imdb and imdb.imdb_rating and imdb.num_votes and imdb.num_votes >= 1000:
        parts.append(f"IMDB score: {imdb.imdb_rating:.1f} from {imdb.num_votes:,} voters.")

    return " ".join(parts)


def load_and_prepare(movielens_dir: Path, imdb_lookup: dict) -> pd.DataFrame:
    movies = pd.read_csv(movielens_dir / "movies.csv")
    tags = pd.read_csv(movielens_dir / "tags.csv")
    ratings = pd.read_csv(movielens_dir / "ratings.csv")
    print(f"[task3] Loaded {len(movies)} movies, {len(tags)} tags, {len(ratings)} ratings")

    # Select up to Top 5 tags per movie by frequency.
    # 5 is was picked randomly common  as a middle ground for tags
    # We can experiment with other number and check the results
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    tag_counts = tags.groupby(["movieId", "tag"]).size().reset_index(name="count")
    top_tags = (
        tag_counts.groupby("movieId")
        .apply(lambda g: g.nlargest(5, "count")["tag"].tolist(), include_groups=False)
        .to_dict()
    )
    
    # Rating stats per movie (avg + count, for the description text)
    rating_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])

    records = []
    enriched = 0
    for _, row in movies.iterrows():
        movie_id = int(row["movieId"])
        title, year = parse_title_year(row["title"])
        genres = parse_genres(row["genres"])
        avg = rating_stats.loc[movie_id, "mean"] if movie_id in rating_stats.index else None
        count = int(rating_stats.loc[movie_id, "count"]) if movie_id in rating_stats.index else 0
        imdb = imdb_lookup.get(movie_id)
        if imdb:
            enriched += 1

        records.append({
            "movie_id": movie_id,
            "title": title,
            "year": year,
            "genres": genres,
            "description": build_description(
                title, year, genres, top_tags.get(movie_id, []), avg, count, imdb
            ),
            # IMDB quality signals are stored in metadata so the router can compute without reading from disk.
            "imdb_rating": imdb.imdb_rating if (imdb and imdb.imdb_rating) else 0.0,
            "imdb_votes": imdb.num_votes if (imdb and imdb.num_votes) else 0,
        })

    df = pd.DataFrame(records)
    print(f"[task3] {enriched}/{len(df)} movies enriched with IMDB")
    print(f"[task3] Sample description:\n  {df.iloc[0]['description']}\n")
    return df


def main():
    imdb_lookup = load_imdb_enrichment(
        settings.imdb_parquet_path,
        settings.movielens_dir / "links.csv",
    )
    df = load_and_prepare(settings.movielens_dir, imdb_lookup)

    print(f"[task3] Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)

    print(f"[task3] Embedding {len(df)} descriptions...")
    embeddings = model.encode(
        df["description"].tolist(),
        batch_size=settings.embedding_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Get the Task 3 store (movies_imdb collection)
    store = get_store("imdb")
    # NOTE this will reset the database if it was previously created. should be commented if we want to keep it as append which we don't.
    store.reset()

    metadatas = [
        {
            "title": r["title"],
            "year": r["year"] if r["year"] is not None else 0,
            "genres": "|".join(r["genres"]),
            "imdb_rating": float(r["imdb_rating"]),
            "imdb_votes": int(r["imdb_votes"]),
        }
        for _, r in df.iterrows()
    ]
    
    # perform upsert insertion supported by chroma. if the db used does not use it. it will throw an error.
    store.upsert_batched(
        ids=df["movie_id"].astype(str).tolist(),
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=df["description"].tolist(),
    )
    print(f"[task3] Done. Collection '{store.collection_name}' has {store.count()} vectors")


if __name__ == "__main__":
    main()