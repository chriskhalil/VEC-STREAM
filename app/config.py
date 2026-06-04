# Default behaviour use .env or defaults to the following Pydantic-settings auto-loads
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Versions we are using for TASK 1. we can add or remove as needed
VERSIONS = ("basic", "imdb")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # path to datasets and vector database
    chroma_db_dir: Path = Path("./data/chroma")
    movielens_dir: Path = Path("./data/movie_lens")
    imdb_parquet_path: Path = Path("./data/imdb/imdb_data.parquet")
    watched_sets_path: Path = Path("./data/watched_sets.pkl") #watched set to keep track who watched what

    # embedding settings
    # WARNING: MAKE SURE THE EMBEDDING MODEL THAT WILL BE USED FITS INTO MEMORY BEFORE DEPLOYING
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 64

    # chromaDB settings 
    # collection names (one per task)
    collection_basic: str = "movies_basic"   # Task 1: MovieLens only
    collection_imdb: str = "movies_imdb"     # Task 3: MovieLens + IMDB

    # Recommendation parameters settings for Task2
    like_threshold: float = 3.5       # ratings >= means it has a good rating
    overfetch_multiplier: int = 4     # fetch the user limit (#needed by user) * N candidates before filtering ( if user watched a movie it should not be recommended)
    overfetch_fallback: int = 1000     # second-pass fetch for heavy watchers

    # Quality-prior and re-ranking settings for Task 3
    quality_prior_weight: float = 0.15   #  0 = pure similarity, 1 = pure quality , how much IMDB rating influence the data.
    min_votes_for_prior: int = 1000      #  required number of votes per film to to use the use the IMDB rating 

    # API  settings 
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # helpers functions
    # keep version-aware path logic in one place. Without them, every script/router would have to know "profiles are at ./data/profiles_{version}.pkl 

    def collection_name(self, version: str) -> str:
        return {"basic": self.collection_basic, "imdb": self.collection_imdb}[version]

    def profile_path(self, version: str) -> Path:
        return Path(f"./data/profiles_{version}.pkl")

    def cold_start_path(self, version: str) -> Path:
        return Path(f"./data/cold_start_{version}.pkl")


settings = Settings()