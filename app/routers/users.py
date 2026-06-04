# Endpoint for Task2
# Task 2 can either use Task1 or Task3 we leave it to the userby wither requesting ?version= (basic or imdb)

from typing import Optional
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Body

from app.config import settings, VERSIONS
from app.schemas import (
    RecommendedMovie, UserRecsMeta, UserRecsResponse,
)
from app.services.vector_store import get_store
from app.services.profile_store import get_profile_store
from app.services.cold_start import get_cold_start
from app.services.watched_store import watched_store

router = APIRouter(prefix="/users", tags=["users"])

def _parse_genres(packed: str) -> list[str]:
    # parse pipe delimeted genra
    return [g for g in packed.split("|") if g] if packed else []


def _confidence(distance: float) -> float:
    # map similarity to [0,1] from [0,2] in chroma
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


def _filter_and_rank(raw: dict, watched: frozenset) -> list[RecommendedMovie]:
    # check the user watched list and remove all movies the user has in is history.
    # map all chroma results to RecommendedeMovie structure.
    out = []
    for cid, meta, dist in zip(raw["ids"][0], raw["metadatas"][0], raw["distances"][0]):
        mid_int = int(cid)
        if mid_int in watched:
            continue
        out.append(RecommendedMovie(
            movie_id=mid_int,
            title=meta["title"],
            year=meta["year"] if meta["year"] else None,
            genres=_parse_genres(meta.get("genres", "")),
            match_confidence=round(_confidence(dist), 4),
        ))
    return out


def _id_to_object(movie_ids: list[int], store) -> list[RecommendedMovie]:
    # Cold start, take movies id and form them into objects
    out = []
    for mid in movie_ids:
        meta = store.get_metadata(str(mid))
        if meta is None:
            continue
        out.append(RecommendedMovie(
            movie_id=mid,
            title=meta["title"],
            year=meta["year"] if meta["year"] else None,
            genres=_parse_genres(meta.get("genres", "")),
            match_confidence=1.0,
        ))
    return out


@router.get("/{user_id}/recommendations", response_model=UserRecsResponse)
def recommendations(
    user_id: int,
    version: str = Query("basic", description="basic = Task 1 vectors, imdb = Task 3 vectors"),
    limit: int = Query(10, ge=1, le=50),
):
    #Personalized recommendations 
    #1. Load persona for this user. If none exists then cold start path.
    #2. Over-fetch candidates from ChromaDB (limit * multiplier).
    #3. Filter out watched movies (O(1) set membership).
    #4. If under-filled (heavy watcher case), fallback to larger fetch.
    #5. Truncate to limit and return.
    
    if version not in VERSIONS:
        raise HTTPException(400, f"version must be one of {VERSIONS}")

    store = get_store(version)
    profiles = get_profile_store(version)
    cold_start = get_cold_start(version)

    persona = profiles.get(user_id)

    # case of unkown user
    if persona is None:
        picks = cold_start.get_recommendations(limit)
        return UserRecsResponse(
            meta=UserRecsMeta(
                user_id=user_id, version=version, limit=limit,
                is_cold_start=True, cold_start_strategy="discovery",
                candidates_fetched=len(picks), candidates_filtered=0,
            ),
            results=_id_to_object(picks, store)[:limit],
        )

    # Personna Exist
    # fetch the personna watch history
    watched = watched_store.get(user_id)

    # First pass: over-fetch so the watched-filter has a safety buffer.
    n = min(limit * settings.overfetch_multiplier, store.count())
    raw = store.query_by_vector(persona.tolist(), n_results=n)
    filtered = _filter_and_rank(raw, watched)

    # if the user has a very big watch history
    # fallback user has watched so many movies our over-fetch wasn't enough.
    if len(filtered) < limit:
        n = min(settings.overfetch_fallback, store.count())
        raw = store.query_by_vector(persona.tolist(), n_results=n)
        filtered = _filter_and_rank(raw, watched)

    return UserRecsResponse(
        meta=UserRecsMeta(
            user_id=user_id, version=version, limit=limit,
            is_cold_start=False,
            candidates_fetched=len(raw["ids"][0]),
            candidates_filtered=len(raw["ids"][0]) - len(filtered),
        ),
        results=filtered[:limit],
    )
