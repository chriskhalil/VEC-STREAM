# 🎬 VEC-STREAM: Production-Grade Media Recommendation Engine

>Recommendation engines power retention at Netflix, Spotify, and Anghami. This project addresses their core challenges from scratch, semantic content understanding, personalized ranking, cold-start handling, and automated content tagging.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-HNSW-orange)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-BGE--small-blueviolet)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063)
![Groq](https://img.shields.io/badge/LLM-Llama--3.1--8b-red)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)

---

## What This Solves

Every major streaming platform: Netflix, Spotify, Anghami, Disney+ and YouTube runs a recommendation engine at its core. The difference between a user who churns in week one and a user who stays for years is often a single well-placed recommendation.

This project implements that engine: semantic understanding of content, personalized user profiles, and intelligent ranking, all served through a production-ready REST API.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Layer                         │
│   GET /items/{id}/similar     GET /users/{id}/recommendations│
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────▼────────────┐
         │      Service Layer      │
         │  VectorStore            │  ← ChromaDB (HNSW, cosine)
         │  ProfileStore           │  ← Pickle → Redis-ready
         │  WatchedStore           │  ← frozenset O(1) lookup
         │  ColdStartCatalog       │  ← k-means + popularity
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │     Offline Pipelines   │
         │  build_basic.py         │  ← Content-only embeddings
         │  build_profiles.py      │  ← User persona vectors
         │  build_enriched.py      │  ← Metadata-enriched embeddings
         │  content_tagger.py      │  ← LLM tag generation + eval
         └─────────────────────────┘
```

**Embedding model:** `BAAI/bge-small-en-v1.5` (384-dim, MTEB retrieval top-tier for its size class)
**Vector database:** ChromaDB with HNSW index, cosine similarity
**API framework:** FastAPI with Pydantic v2 schemas
**LLM backend:** Llama-3.1-8b-instant via Groq API

---

## Core Capabilities

### 1 — Content-Based Similarity
*"Users who liked this also liked..."*

Given any item ID, returns the top-N most semantically similar items using dense vector embeddings over structured natural-language descriptions. This is the backbone of Spotify's "Song Radio" and Netflix's "More Like This."

### 2 — Personalized User Recommendations
*"Picked for you"*

Builds a **rating-weighted taste profile** (persona vector) per user from their interaction history. Recommendations are retrieved by nearest-neighbor search in that personal taste space — the same approach used in Anghami's personalized playlists and Netflix's homepage rows.

Includes:
- Watched/consumed item filtering with O(1) set lookup, no re-recommendations ever
- Adaptive overfetching for power users with large histories
- **Cold-start handling** for new users (k-means diversity sampling + popularity backstop)

### 3 — Quality-Aware Re-Ranking
*"Similar AND worth watching"*

Enhances pure similarity with an external quality signal (ratings, votes) via a tunable blending formula:

```
rank_score = (1 - w) × semantic_similarity + w × quality_prior
```

This mirrors how Netflix blends predicted rating with predicted engagement, and how Spotify weights editorial quality into algorithmic playlists. The weight `w` is fully configurable, `0` is pure similarity, `1` is pure quality.

### 4 — LLM Content Tagging
*"Auto-tagging at scale"*

A pipeline that uses a large language model to generate descriptive tags from item descriptions (plot summaries, artist bios, show synopses), then evaluates them against ground-truth labels using both lexical and semantic metrics. This addresses a real operational challenge at scale: Anghami and Spotify ingest thousands of new tracks weekly that need metadata before any embedding can be generated.

---

## Quickstart

### Prerequisites
```bash
python 3.10+
pip install -r requirements.txt
```

### 1. Prepare your dataset

The engine is **dataset-agnostic**. Bring your own catalog: movies, songs, podcasts, shows. You need:

```
data/
├── items.csv       # item_id, title, genres/categories
├── ratings.csv     # user_id, item_id, rating, timestamp
├── tags.csv        # user_id, item_id, tag
└── links.csv       # item_id ↔ external_metadata_id  (optional, for enrichment)
```

### 2. Build the vector indexes

```bash
# Content-only embeddings
python -m scripts.build_basic

# Metadata-enriched embeddings (requires external metadata source)
python -m scripts.build_enriched
```

### 3. Build user profiles

```bash
python -m scripts.build_profiles --version basic
python -m scripts.build_profiles --version enriched
```

### 4. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs: `http://localhost:8000/docs`

---

## API Reference

### Similar Items

```bash
# Content-based similarity
curl "http://localhost:8000/items/1/similar?version=basic&limit=10"

# Quality-aware similarity
curl "http://localhost:8000/items/1/similar?version=enriched&limit=10"
```

**Response:**
```json
{
  "meta": {
    "item_id": 1,
    "query_title": "Inception",
    "version": "enriched",
    "limit": 10
  },
  "results": [
    {
      "item_id": 4226,
      "title": "Memento",
      "year": 2000,
      "genres": ["Mystery", "Thriller"],
      "similarity_score": 0.9314
    },
    {
      "item_id": 7153,
      "title": "The Dark Knight",
      "year": 2008,
      "genres": ["Action", "Crime", "Drama"],
      "similarity_score": 0.9102
    }
  ]
}
```

### Personalized Recommendations

```bash
# Recommendations for a known user
curl "http://localhost:8000/users/42/recommendations?version=enriched&limit=10"

# New user — automatic cold-start
curl "http://localhost:8000/users/999999/recommendations?version=basic&limit=10"
```

**Response:**
```json
{
  "meta": {
    "user_id": 42,
    "version": "enriched",
    "limit": 10,
    "is_cold_start": false,
    "candidates_fetched": 40,
    "candidates_filtered": 12
  },
  "results": [
    {
      "item_id": 296,
      "title": "Pulp Fiction",
      "year": 1994,
      "genres": ["Comedy", "Crime", "Thriller"],
      "match_confidence": 0.8734
    }
  ]
}
```

### Health Check

```bash
curl "http://localhost:8000/health"
```

```json
{
  "status": "ok",
  "basic":    { "items_indexed": 9742, "user_profiles": 610 },
  "enriched": { "items_indexed": 9742, "user_profiles": 610 },
  "consumed_sets": 610
}
```

---

## Project Structure

```
.
├── app/
│   ├── main.py              # FastAPI entrypoint, startup warmup
│   ├── config.py            # All tuneable parameters via env / .env
│   ├── schemas.py           # Request/response models (Pydantic v2)
│   └── routers/
│       ├── items.py         # /items endpoints
│       └── users.py         # /users endpoints
│   └── services/
│       ├── vector_store.py  # ChromaDB adapter (swappable)
│       ├── profile_store.py # User persona store (swap → Redis)
│       ├── watched_store.py # Consumed-item store
│       ├── cold_start.py    # Cold-start catalog
│       └── metadata_loader.py  # External metadata → catalog join
├── scripts/
│   ├── build_basic.py       # Build content-only index
│   ├── build_profiles.py    # Build user profiles + cold-start
│   └── build_enriched.py   # Build metadata-enriched index
├── content_tagger.py        # LLM tagging pipeline + evaluation
└── data/                    # Your dataset goes here
```

---

## Configuration

Zero code changes needed — tune everything via `.env`:

```env
# Embedding
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_BATCH_SIZE=64

# Recommendation
LIKE_THRESHOLD=3.5           # Minimum rating to count as positive signal
OVERFETCH_MULTIPLIER=4       # Candidate buffer before consumed-filter
QUALITY_PRIOR_WEIGHT=0.15    # 0 = pure similarity, 1 = pure quality
MIN_VOTES_FOR_PRIOR=1000     # Vote threshold to trust quality signal

# API
API_HOST=0.0.0.0
API_PORT=8000
```

---

## Design Decisions

**Why semantic embeddings over collaborative filtering?**  
Collaborative filtering requires a dense interaction matrix — it collapses on new items and sparse users. Semantic embeddings generalize from content descriptions, enabling recommendations the moment an item is ingested with zero interaction data. This is why Spotify uses embeddings for new track discovery and Netflix for new original titles.

**Why BGE-small over larger models?**  
`bge-small-en-v1.5` ranks at the top of the MTEB retrieval leaderboard for sub-100M parameter models. It fits in CPU memory (~500MB), making it deployable without GPU infrastructure. The next step up would be `bge-large` or `e5-mistral-7b` for production systems with GPU budget.

**Why rating-weighted persona vectors?**  
A simple average of a user's item embeddings ignores the strength of preference. Weighting by `rating - threshold` means a 5-star item pulls the persona harder than a 3.5-star item — capturing taste intensity, not just taste direction. This is the core idea behind weighted matrix factorization used at Netflix and Spotify.

**Why overfetch-then-filter?**  
Pushing consumed-item filtering to the application layer keeps the hot path simple and avoids metadata filter overhead on every vector query. For power users with very large histories, the fallback pass and, in production, native vector DB exclusion filters (Qdrant supports this natively) handle the edge case cleanly.

**Why a two-pronged cold start?**  
K-means cluster centroids cover taste-space diversity; the popularity list adds a quality backstop. Neither alone is sufficient — pure popularity is boring, pure diversity risks surfacing obscure content to a user who just wants something reliable.

---

## Production Roadmap

| Current State | Production Upgrade |
|---|---|
| No response cache | Redis LRU on `(item_id, version, limit)` |
| Pickle persistence | Redis for consumed-sets, Protobuf for profiles |
| Nightly batch profile rebuild | Incremental persona update per rating event |
| Single ChromaDB instance | Qdrant / Milvus with sharded HNSW |
| No test suite | pytest: golden-file + property-based tests |
| Consumed-set = rated items | Separate event streams: `play_started`, `completed`, `rated`, `skipped` |

---

## LLM Tagging Evaluation

The tagging pipeline evaluates generated tags against ground-truth labels across both lexical and semantic dimensions:

| Metric | What It Measures |
|---|---|
| **F1 / Precision / Recall** | Lexical token overlap with ground truth |
| **Jaccard** | Set-level similarity |
| **Novelty Rate** | Tags beyond the ground-truth vocabulary — desirable for discovery |
| **sem_avg_max_cosine** | Per-tag semantic similarity to nearest ground-truth tag |
| **sem_set_cosine** | Centroid-level semantic alignment of tag clouds |

> Lexical metrics understate LLM quality on noisy, user-generated ground truth. Semantic metrics (`sem_avg_max_cosine`) are the more reliable signal for this corpus.

Results are written to `data/output_llm/evaluation_metrics.json` after running the pipeline.

---

## Relevance to Industry

| Platform | This System Addresses |
|---|---|
| **Netflix** | "More Like This", personalized homepage rows, cold-start for new originals |
| **Spotify** | Song Radio, Discover Weekly persona vectors, auto-tagging new track uploads |
| **Anghami** | Personalized playlists, new user onboarding, Arabic content cold-start |
| **YouTube** | Watch-history filtering, quality-aware re-ranking, semantic tag generation |
| **Disney+** | Cross-catalog similarity (movie ↔ series), franchise-aware recommendations |

---

*Built as a deep-dive into production recommendation systems. Dataset-agnostic — plug in any catalog.*
