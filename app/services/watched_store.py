# Create a watch list for each user.
# we do not want at any cost recommend a video that the user interacted with be it watched/ semi-watched / liked or disliked
# this class keep tracks of user watch history and filters the chroma results.
# frozenset is used to have a readonly O(1) lookup table.
 
from __future__ import annotations
import pickle
from pathlib import Path


class WatchedStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[int, frozenset[int]] = {}

    def load(self) -> "WatchedStore":
        if self.path.exists():
            with open(self.path, "rb") as f:
                self._data = pickle.load(f)
        return self

    def save(self) -> "WatchedStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return self

    def replace(self, data: dict[int, frozenset[int]]) -> "WatchedStore":
        self._data = data
        return self

    def get(self, user_id: int) -> frozenset[int]:
        #Returns empty frozenset for unknown users
        return self._data.get(user_id, frozenset())

    def set(self, user_id: int, movie_ids) -> None:
        self._data[user_id] = frozenset(movie_ids)

    def __len__(self) -> int:
        return len(self._data)

from app.config import settings
watched_store = WatchedStore(settings.watched_sets_path)