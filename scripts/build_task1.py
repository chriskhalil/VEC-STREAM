# TASK 1 — Build content-based recommender using MovieLens data.

#Pipeline:
#1. Read movies.csv, tags.csv, ratings.csv
#2. For each movie, compose a natural-language description from  title, year, genres, top tags, and rating stats
#3. Embed with BGE-small-en-v1.5
#4. Upsert into ChromaDB collection 'movies_basic'

#Run: python -m scripts.build_task1

import re
import sys
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.vector_store import get_store

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
) -> str:

    # NOTE
    # here we will createa vectors embedding for searching.
    # to be able to work with accurate vectors we need to append our data we extracted into well formed english sentence.
    # for this i will use a generate sentence template. BUT to get better results you can use llms to generate a factual well sentences that are more descriptive.
    # This is needed because embedding vectors are trained on actual english sentences and not words.
    # this means that usina someting like  "{movie} {year}. Genres:{genra}. Viewers describe it as  {rating}."
    # this is very basic but can be enhanced with LLMS

    parts = [f"{title} ({year})." if year else f"{title}."]

    if genres:
        parts.append(
            f"A {genres[0]} film." if len(genres) == 1
            else f"Genres: {', '.join(genres)}."
        )

    if top_tags:
        parts.append(f"Viewers describe it as {', '.join(top_tags)}.")

    if avg_rating is not None and num_ratings >= 5:
        parts.append(f"Rated {avg_rating:.1f} out of 5 by {num_ratings} viewers.")

    return " ".join(parts)


def load_and_prepare(movielens_dir: Path) -> pd.DataFrame:#
    # load and process MOvie lens dataset and add a desription for each movue
    movies = pd.read_csv(movielens_dir / "movies.csv")
    tags = pd.read_csv(movielens_dir / "tags.csv")
    ratings = pd.read_csv(movielens_dir / "ratings.csv")
    print(f"[task1] Loaded {len(movies)} movies, {len(tags)} tags, {len(ratings)} ratings")

    # select up to Top 5 tags per movie by frequency.
    #  5 is was picked randomly common  as a middle ground for tags
    #  we can experiment with other number and check the results
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
    for _, row in movies.iterrows():
        movie_id = int(row["movieId"])
        title, year = parse_title_year(row["title"])
        genres = parse_genres(row["genres"])
        avg = rating_stats.loc[movie_id, "mean"] if movie_id in rating_stats.index else None
        count = int(rating_stats.loc[movie_id, "count"]) if movie_id in rating_stats.index else 0

        records.append({
            "movie_id": movie_id,
            "title": title,
            "year": year,
            "genres": genres,
            "description": build_description(
                title, year, genres, top_tags.get(movie_id, []), avg, count
            ),
        })

    df = pd.DataFrame(records)
    print(f"[task1] Sample description:\n  {df.iloc[0]['description']}\n")
    return df


def main():
    df = load_and_prepare(settings.movielens_dir)

    print(f"[task1] Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)

    # normalize_embeddings=True: BGE models expect L2-normalized output
    # for cosine similarity to behave correctly.
    print(f"[task1] Embedding {len(df)} descriptions...")
    embeddings = model.encode(
        df["description"].tolist(),
        batch_size=settings.embedding_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"[task1] Embeddings shape: {embeddings.shape}")

    # Get the Task 1 store (movies_basic collection) and reset for clean rebuild
    store = get_store("basic")
    # NOTE this will reset the database if it was previously created. should be commented if we want to keep it as append which we don't.
    store.reset()

    metadatas = [
        {
            "title": r["title"],
            "year": r["year"] if r["year"] is not None else 0,  # Chroma requires scalar
            "genres": "|".join(r["genres"]),                     # pipe-delimited for roundtrip
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
    print(f"[task1] Done. Collection '{store.collection_name}' has {store.count()} vectors")


if __name__ == "__main__":
    main()