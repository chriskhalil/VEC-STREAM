#  ColdStartCatalog to initiate a new user.  this was not asked specifically in task2 but it is a real problem for recommender systems.
#  We can initiate a cold start in multiple ways. this class hold a precomputed catalog computed offline.
#  note without this any new user will have an empty set of recommendation as we do not have the user embbedinng vector.
#
# we use two type of recommendation:
# 1. give K most popular movies.
# 2. give K representative movies for genres we have from our dataset 

from __future__ import annotations
import pickle
from pathlib import Path


class ColdStartCatalog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, list[int]] = {"discovery": [], "popular": []}

    def load(self) -> "ColdStartCatalog":
        if self.path.exists():
            with open(self.path, "rb") as f:
                self._data = pickle.load(f)
        return self

    def save(self, discovery: list[int], popular: list[int]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = {"discovery": discovery, "popular": popular}
        with open(self.path, "wb") as f:
            pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_recommendations(self, limit: int) -> list[int]:
        """Interleave discovery + popular, dedup, truncate.

        Discovery comes first because diversity is more valuable than
        popularity for a genuinely unknown user.
        """
        seen, out = set(), []
        for mid in self._data["discovery"] + self._data["popular"]:
            if mid not in seen:
                seen.add(mid)
                out.append(mid)
            if len(out) >= limit:
                break
        return out


from app.config import settings

cold_start_basic = ColdStartCatalog(settings.cold_start_path("basic"))
cold_start_imdb = ColdStartCatalog(settings.cold_start_path("imdb"))


def get_cold_start(version: str) -> ColdStartCatalog:
    if version == "basic":
        return cold_start_basic
    if version == "imdb":
        return cold_start_imdb
    raise ValueError(f"unknown version: {version}")