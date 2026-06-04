from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass, field

# classes for Task 1 and Task 3

class SimilarMovie(BaseModel):
    movie_id: int
    title: str
    year: Optional[int] = None
    genres: List[str]
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class SimilarMoviesMeta(BaseModel):
    movie_id: int
    query_title: str
    #version: "basic" (Task 1) or "imdb" (Task 3)
    version: str             
    limit: int


class SimilarMoviesResponse(BaseModel):
    meta: SimilarMoviesMeta
    results: List[SimilarMovie]


class ImdbRecord(BaseModel):
    plot: Optional[str] = None  ## after inspecting the parquet file some plots may be missing
    directors: list[str] = field(default_factory=list)
    cast: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    imdb_rating: Optional[float] = None
    num_votes: Optional[int] = None
    runtime_minutes: Optional[int] = None

# classes for Task 2

class RecommendedMovie(BaseModel):
    movie_id: int
    title: str
    year: Optional[int] = None
    genres: List[str]
    match_confidence: float = Field(..., ge=0.0, le=1.0)


class UserRecsMeta(BaseModel):
    user_id: int
    version: str              # which vector space was used
    limit: int
    is_cold_start: bool = False
    cold_start_strategy: Optional[str] = None   # "seeds" | "discovery" | None
    
    # Observability: lets us see the watched-filter behavior during stress tests
    candidates_fetched: int
    candidates_filtered: int


class UserRecsResponse(BaseModel):
    meta: UserRecsMeta
    results: List[RecommendedMovie]

