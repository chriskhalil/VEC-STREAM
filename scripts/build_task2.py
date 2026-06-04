 # TASK 2 — User suggestions
 # Each user profile is dependent of the TASK used as backend if TASK1 or TASK3
 # a watched movie is watched regardles off the TASK

 #Pipeline:
  #1. Load ratings.csv
  #2. For each user: rating-weighted persona from their liked movies vectors 
  #3. For each user: watched-set (all rated movies, any rating)
  #4. Cold-start catalog: k-means on movie vectors for diversity +  popularity list for quality backstop
  #5. Persist to pickles

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings, VERSIONS
from app.services.vector_store import get_store
from app.services.profile_store import ProfileStore
from app.services.watched_store import WatchedStore
from app.services.cold_start import ColdStartCatalog


def load_movie_vectors(store):

    # Pull all embending from chromaDB collection and dump it into memory.
    # this is needed for K-means as readin and writing to disk is slow.
    # this is not good optimized way check for redis or other in-memory db

    raw = store.collection.get(include=["embeddings"])
    ids = [int(x) for x in raw["ids"]]
    vecs = np.asarray(raw["embeddings"], dtype=np.float32)
    id_to_vec = {mid: vecs[i] for i, mid in enumerate(ids)}
    return id_to_vec, ids, vecs


def build_user_profiles(ratings, id_to_vec, like_threshold):
    # Crating Rating-weighted persona vectors.
    # Formula: persona_u = Σ(w_i * v_i) / Σ(w_i) where w_i = rating_i - (threshold - 0.5)
    # check :https://link.springer.com/book/10.1007/978-0-387-85820-3

   # The -0.5 shift means a 3.5-star rating gets weight 0.5 (counts lightly) rather than 0 (ignored). Smoother than a hard threshold.
   # we need to wright based of context. 5 start with love should have greater impact than a 3.5 start with decent tag.

    profiles = {}
    shift = like_threshold - 0.5

    for user_id, group in ratings.groupby("userId"):
        weights, vectors = [], []
        for _, row in group.iterrows():
            if row["rating"] < like_threshold:
                continue  # negative signal — skip entirely
            mid = int(row["movieId"])
            if mid in id_to_vec:
                weights.append(row["rating"] - shift)
                vectors.append(id_to_vec[mid])

        if not vectors:
            continue  # user had no likes above threshold

        W = np.asarray(weights, dtype=np.float32)
        V = np.stack(vectors)
        persona = (W[:, None] * V).sum(axis=0) / W.sum()

        # L2-normalize so cosine similarity is consistent with movie vectors
        norm = np.linalg.norm(persona)
        if norm > 0:
            persona = persona / norm
        profiles[int(user_id)] = persona.astype(np.float32)

    return profiles


def build_watched_sets(ratings):
    # use the following logic if a user rated a movie this means the user watched the movie.
    # we assume that  rating == watched
    return {
        int(uid): frozenset(g["movieId"].astype(int).tolist())
        for uid, g in ratings.groupby("userId")
    }


def build_cold_start(ordered_ids, matrix, ratings, k=8, per_list=15):
    #Two-pronged cold start.

    #discovery: k-means on the movie vector space; one pick per cluster centroid. Covers taste-space diversity
    #popular:   highest avg rating with >= 50 votes. 
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(matrix)
    discovery = []
    for c in range(k):
        # For each cluster, pick the movie closest to the centroid
        dists = np.linalg.norm(matrix - km.cluster_centers_[c], axis=1)
        discovery.append(ordered_ids[int(np.argmin(dists))])

    stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    popular = (
        stats[stats["count"] >= 50]
        .sort_values("mean", ascending=False)
        .head(per_list).index.astype(int).tolist()
    )
    return discovery, popular


def main():
    parser = argparse.ArgumentParser(description="Build Task 2 artifacts for a given task version.")
    parser.add_argument(
        "--version", choices=VERSIONS, required=True,
        help="Which movie vector space to build profiles against",
    )
    args = parser.parse_args()
    version = args.version

    ratings = pd.read_csv(settings.movielens_dir / "ratings.csv")
    print(f"[task2/{version}] Loaded {len(ratings)} ratings from {ratings['userId'].nunique()} users")

    # Read from whichever collection the user asked for
    store = get_store(version)
    if store.count() == 0:
        build_script = "build_task1" if version == "basic" else "build_task3"
        raise RuntimeError(
            f"Collection '{store.collection_name}' is empty. "
            f"Run `python -m scripts.{build_script}` first."
        )

    id_to_vec, ordered_ids, matrix = load_movie_vectors(store)
    print(f"[task2/{version}] Loaded {len(id_to_vec)} movie vectors")

    profiles = build_user_profiles(ratings, id_to_vec, settings.like_threshold)
    print(f"[task2/{version}] Built {len(profiles)} user profiles")

    watched = build_watched_sets(ratings)
    print(f"[task2/{version}] Built {len(watched)} watched sets")

    discovery, popular = build_cold_start(ordered_ids, matrix, ratings)
    print(f"[task2/{version}] Cold-start discovery picks: {discovery}")
    print(f"[task2/{version}] Cold-start popular (first 5): {popular[:5]}")

    # Persist everything.
    # Profiles and cold-start are version-specific;
    # watched sets are shared across versions.
    ProfileStore(settings.profile_path(version)).replace(profiles).save()
    WatchedStore(settings.watched_sets_path).replace(watched).save()
    ColdStartCatalog(settings.cold_start_path(version)).save(discovery, popular)
    print(f"[task2/{version}] Saved all artifacts")


if __name__ == "__main__":
    main()