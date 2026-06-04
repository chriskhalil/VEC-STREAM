#FastAPI application entrypoint.
#All real logic lives in routers/ and services/.

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import VERSIONS
from app.services.vector_store import get_store
from app.services.profile_store import get_profile_store
from app.services.cold_start import get_cold_start
from app.services.watched_store import watched_store
from app.routers import movies, users


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm every store on startup so the first real request is fast.

    for v in VERSIONS:
        store = get_store(v)
        profiles = get_profile_store(v).load()
        cold = get_cold_start(v).load()
        print(
            f"[startup/{v}] movies={store.count()} "
            f"profiles={len(profiles)} "
            f"cold_start_ready={bool(cold._data['discovery'])}"
        )

    watched_store.load()
    print(f"[startup] watched_sets={len(watched_store)}")

    # Warn loudly if any task hasn't been built yet.
    for v in VERSIONS:
        if get_store(v).count() == 0:
            script = "build_task1" if v == "basic" else "build_task3"
            print(f"[startup] WARNING: '{v}' collection is empty. Run: python -m scripts.{script}")
        if len(get_profile_store(v)) == 0:
            print(f"[startup] WARNING: no '{v}' profiles. Run: python -m scripts.build_task2 --version {v}")

    yield
    print("[shutdown] Goodbye.")


app = FastAPI(
    title="MovieLens Recommender - Christophe @CK",
    description="Tasks 1, 2, 3 — independent, comparable via ?version=",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(movies.router)
app.include_router(users.router)


@app.get("/health", tags=["ops"])
def health():
    return {
        "status": "ok",
        "basic": {
            "movies_indexed": get_store("basic").count(),
            "user_profiles": len(get_profile_store("basic")),
        },
        "imdb": {
            "movies_indexed": get_store("imdb").count(),
            "user_profiles": len(get_profile_store("imdb")),
        },
        "watched_sets": len(watched_store),
    }