# End point for TASK 1 and 3
# as both task perform the same task we combine. and add the ?version=basic for task 1 and ?version=imdb for task2

from fastapi import APIRouter, HTTPException, Query
from app.config import settings, VERSIONS
from app.schemas import SimilarMovie, SimilarMoviesMeta, SimilarMoviesResponse
from app.services.vector_store import get_store

router = APIRouter(prefix="/movies", tags=["movies"])


def _parse_genres(packed: str) -> list[str]:
    # parse pipe delimeted genra
    return [g for g in packed.split("|") if g] if packed else []


def _similarity(distance: float) -> float:
    # map similarity to [0,1] from [0,2] in chroma
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


def _quality_prior(imdb_rating: float, imdb_votes: int) -> float:
    # normalize  IMDB rating to [0,1] -> do not boost if voters less than min_votes_for_prior
    if imdb_votes < settings.min_votes_for_prior:
        return 0.0
    return max(0.0, min(1.0, imdb_rating / 10.0))


@router.get("/{movie_id}/similar", response_model=SimilarMoviesResponse)
def similar_movies(
    movie_id: int,
    version: str = Query("basic", description="basic = Task 1, imdb = Task 3"),
    limit: int = Query(10, ge=1, le=50),
):
    # Return top-N similar movies using semantic embeddings.
    #1. Look up the query movie's embedding from the chosen collection.
    #2. HNSW nearest-neighbor search (over-fetch to allow re-ranking).
    #3. If imdb version: blend similarity with quality prior.
    #4. Drop self-match, sort, truncate.


    # NOTE : THIS SHOULD HAVE A CACHE LAYER INSERTED HERE. I DID NOT IMPLEMENT IT FOR THIS TEST PROJECT. 
    #        HAVING A CACHE LAYER HERE WILL REDUCE COMPUTATION STRESS ON OUR SERVER BY DIRECTLY SERVING SIMILAR QUERIES. IF THE SYSTEM WILL BE IMPLEMENTED IN PRODUCTION IT IS A MUST
    
    
    #check version mismatch
    if version not in VERSIONS:
        raise HTTPException(400, f"version must be one of {VERSIONS}")


    #load store for our version either imdb or basic
    store = get_store(version)
    mid = str(movie_id)

    # get the metadata from the following movie.
    query_meta = store.get_metadata(mid)

    if query_meta is None:
        raise HTTPException(404, f"movie_id {movie_id} not found in '{version}' index")

    query_vec = store.get_embedding(mid)

   
   
    # Overfetching to remove our movie cannot recommend the same movie.

    overfetch = min(limit * 3 + 1, store.count())
    raw = store.query_by_vector(query_vec, n_results=overfetch)

    ids = raw["ids"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    # Task 3 applies quality prior; Task 1 uses pure similarity.
    use_prior = (version == "imdb")
    w = settings.quality_prior_weight

    scored = []
    for cid, meta, dist in zip(ids, metadatas, distances):
        if cid == mid:
            continue  # drop self-match 
        sim = _similarity(dist)
        if use_prior:
            prior = _quality_prior(
                float(meta.get("imdb_rating", 0.0)),
                int(meta.get("imdb_votes", 0)),
            )
            rank_score = (1 - w) * sim + w * prior   ## give the weights for the prior  ## must add to 1
        else:
            rank_score = sim
        scored.append((rank_score, sim, cid, meta))

    # Sort by blended score, but report the pure similarity in the response
    # (blended score conflates "similar" with "good" this imploes that the user become confused).
    scored.sort(key=lambda x: x[0], reverse=True)

    results = [
        SimilarMovie(
            movie_id=int(cid),
            title=meta["title"],
            year=meta["year"] if meta["year"] else None,
            genres=_parse_genres(meta.get("genres", "")),
            similarity_score=round(sim, 4),
        )
        for _, sim, cid, meta in scored[:limit]
    ]

    return SimilarMoviesResponse(
        meta=SimilarMoviesMeta(
            movie_id=movie_id,
            query_title=query_meta["title"],
            version=version,
            limit=limit,
        ),
        results=results,
    )