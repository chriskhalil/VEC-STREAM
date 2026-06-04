# This file is an interface Profile Store for users ( user preference embedding) 
# It was created as such for the toy dataset provided.
# For production use Redis or Aerospike
# This implementation has O(1) lookups and normal reading from disk at service start.
# vectors are expensive to compute but cheap to store we compute them each night offline to update.


# we picked pickle over json and sql because it is nately serializable for numpy array

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Optional
import numpy as np


class ProfileStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[int, np.ndarray] = {}


    #
    # CRUD operations. Note that Updates here is named replace and will replace the full embedding
    #

    def load(self) -> "ProfileStore":
        #
        # fetch and load pickle file. return ProfileStore for chaining
        if self.path.exists():
            with open(self.path, "rb") as f:
                self._data = pickle.load(f)
        return self

    def save(self) -> "ProfileStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL) # HIGHEST_PROTOCOL to ensure the most efficient save per documentation hwo it differs from normall no idea
        return self

    def replace(self, data: dict[int, np.ndarray]) -> "ProfileStore":
        """Bulk-replace internal dict. Used by build scripts."""
        self._data = data
        return self

    def get(self, user_id: int) -> Optional[np.ndarray]:
        return self._data.get(user_id)

    def set(self, user_id: int, vector: np.ndarray) -> None:
        # Note vector is normalized using L2

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        self._data[user_id] = vector.astype(np.float32)

    def __len__(self) -> int:
        return len(self._data)


from app.config import settings

profile_store_basic = ProfileStore(settings.profile_path("basic"))
profile_store_imdb = ProfileStore(settings.profile_path("imdb"))


def get_profile_store(version: str) -> ProfileStore:
    """Route to the right profile store by task version."""
    if version == "basic":
        return profile_store_basic
    if version == "imdb":
        return profile_store_imdb
    raise ValueError(f"unknown version: {version}")