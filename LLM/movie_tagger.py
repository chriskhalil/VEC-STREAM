from __future__ import annotations

import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Dependencies
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False
    print("sentence-transformers not installed - semantic metrics skipped.")

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    print(" groq SDK not installed.")


# Data structures
@dataclass
class MovieRecord:
    movie_id: int
    title: str
    plot: str
    genres: List[str]
    ground_truth_tags: List[str]


@dataclass
class TaggingResult:
    movie_id: int
    title: str
    predicted_tags: List[str]
    confidences: Dict[str, float]
    ground_truth_tags: List[str]
    raw_llm_output: str
    latency_ms: float


@dataclass
class EvaluationResult:
    movie_id: int
    jaccard: float
    precision: float
    recall: float
    f1: float
    novelty_rate: float
    exact_ground_truth_coverage: float
    sem_avg_max_cosine: Optional[float] = None
    sem_set_cosine: Optional[float] = None


def load_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


class MovieTaggerLLM:
    # we are using the groq the API as it provides free tokens.

    def __init__(
        self,
        system_prompt: str,
        model_name: str = "llama-3.1-8b-instant",
        max_retries: int = 3,
        backoff_base: float = 1.5,
    ):
        if not _GROQ_AVAILABLE:
            raise ImportError("Install groq: pip install groq")
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def generate_tags(self, plot: str) -> Tuple[List[str], Dict[str, float], str, float]:
        #Returns (tags, confidences, raw_output, latency_ms)
        if not plot or (isinstance(plot, float) and pd.isna(plot)):
            return [], {}, "", 0.0

        raw, latency = "", 0.0
        for attempt in range(self.max_retries):
            try:
                t0 = time.perf_counter()
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Plot: {plot.strip()}\n"},
                    ],
                    model=self.model_name,
                    temperature=0,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                )
                latency = (time.perf_counter() - t0) * 1000
                raw = response.choices[0].message.content.strip()
                tags, confs = self._parse_json(raw)

                if len(tags) >= 2:
                    return tags, confs, raw, latency

                print(f"got {len(tags)} tag(s), retrying...")

            except Exception as exc:
                wait = self.backoff_base ** attempt
                print(f"error: {exc} - waiting {wait:.1f}s")
                time.sleep(wait)

        return [], {}, raw, latency

    @staticmethod
    def _parse_json(text: str) -> Tuple[List[str], Dict[str, float]]:

        #Parses JSON of shape: {"tags": [{"tag": "wrestling", "confidence": 0.95}, ...]}
        #     LLM bla bla bla


        # Extract the first JSON object. unmatch any talking the llm does
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return [], {}

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [], {}

        items = data.get("tags", [])
        if not isinstance(items, list):
            return [], {}

        tags, confs = [], {}
        for item in items[:5]:  # hard cap at 5
            if isinstance(item, dict):
                tag = str(item.get("tag", "")).strip().lower()
                conf = float(item.get("confidence", 0.5))
            elif isinstance(item, str):
                tag = item.strip().lower()
                conf = 0.5
            else:
                continue

            if tag:
                tags.append(tag)
                confs[tag] = min(1.0, max(0.0, conf))

        return tags, confs


# Evaluator
class TagEvaluator:

    def __init__(self, sbert_model_name: str = "all-MiniLM-L6-v2"):
        self._sbert: Optional[SentenceTransformer] = None
        if _SBERT_AVAILABLE:
            print(f"Loading SBERT: {sbert_model_name}...")
            self._sbert = SentenceTransformer(sbert_model_name)

    @staticmethod
    def _tokenize(tags: List[str]) -> set:
        tokens = set()
        for t in tags:
            tokens.update(re.findall(r'\b\w+\b', t.lower()))
        return tokens

    def evaluate(self, movie_id: int, predicted: List[str], ground_truth: List[str]) -> EvaluationResult:
        # Lexical
        p_tok = self._tokenize(predicted)
        g_tok = self._tokenize(ground_truth)

        if not p_tok or not g_tok:
            jaccard = precision = recall = f1 = 0.0
        else:
            inter = p_tok & g_tok
            # Precision: What % of our predictions were actually in the ground truth?
            precision = len(inter) / len(p_tok)
            recall = len(inter) / len(g_tok)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            jaccard = len(inter) / len(p_tok | g_tok)

        # Coverage
        if predicted:
            ground_truth_lower = {g.lower() for g in ground_truth}
            pred_lower = [p.lower() for p in predicted]
            novelty = sum(1 for p in pred_lower if p not in ground_truth_lower) / len(predicted)
            exact = (sum(1 for g in ground_truth_lower if g in pred_lower) / len(ground_truth_lower)) if ground_truth_lower else 0.0
        else:
            novelty = exact = 0.0

        # Set Cosine: Mean pooling of all vectors in each set, then calculating similarity.
        # Represents the "centroid" similarity of the two tag clouds.
        # Normalize vectors to unit lenground_truthh for cosine similarity calculation
        avg_max = set_cos = None
        if self._sbert and predicted and ground_truth:
            pe = self._sbert.encode(predicted, convert_to_numpy=True, show_progress_bar=False)
            ge = self._sbert.encode(ground_truth, convert_to_numpy=True, show_progress_bar=False)
            pe = pe / (np.linalg.norm(pe, axis=1, keepdims=True) + 1e-10)
            ge = ge / (np.linalg.norm(ge, axis=1, keepdims=True) + 1e-10)
            avg_max = round(float(np.mean((pe @ ge.T).max(axis=1))), 4)
            set_cos = round(float(pe.mean(axis=0) @ ge.mean(axis=0)), 4)

        return EvaluationResult(
            movie_id=movie_id,
            jaccard=round(jaccard, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            novelty_rate=round(novelty, 4),
            exact_ground_truth_coverage=round(exact, 4),
            sem_avg_max_cosine=avg_max,
            sem_set_cosine=set_cos,
        )


# Data loading

def load_ground_truth_tags(tags_csv: Path) -> Dict[int, List[str]]:
    df = pd.read_csv(tags_csv)
    return (
        df.groupby("movieId")["tag"]
        .apply(lambda x: list(set(x.dropna().astype(str).str.lower())))
        .to_dict()
    )


def load_movie_metadata(movies_csv: Path) -> Dict[int, Tuple[str, List[str]]]:
    df = pd.read_csv(movies_csv)
    result = {}
    for _, row in df.iterrows():
        genres = [
            g.strip() for g in str(row.get("genres", "")).split("|")
            if g.strip() and g.strip() != "(no genres listed)"
        ]
        result[int(row["movieId"])] = (str(row.get("title", "")), genres)
    return result


def load_imdb_plots(imdb_parquet: Path, links_csv: Path) -> Dict[int, str]:
    links = pd.read_csv(links_csv)
    imdb = pd.read_parquet(imdb_parquet)
    merged = links.merge(imdb, on="imdbId", how="inner")

    plot_col = next(
        (c for c in merged.columns if "plot" in c.lower() or "summary" in c.lower()),
        None,
    )
    if plot_col is None:
        raise ValueError(f"No plot column in IMDB parquet. Columns: {list(imdb.columns)}")

    return dict(zip(merged["movieId"].astype(int), merged[plot_col].astype(str)))


def stratified_sample(records: List[MovieRecord], n: int, seed: int = 42) -> List[MovieRecord]:
    if n >= len(records):
        return records

    rng = np.random.default_rng(seed)
    # Group records into "buckets" based on their primary genre
    buckets: Dict[str, List[MovieRecord]] = {}
    for r in records:
        key = r.genres[0] if r.genres else "Unknown"
        buckets.setdefault(key, []).append(r)
   
    # Determine how many items to pull from each bucket to reach the target 'n'
    per_genre = max(1, n // len(buckets))
    sampled: List[MovieRecord] = []
    for pool in buckets.values():
        # Ensure we don't try to sample more items than exist in a specific bucket
        k = min(per_genre, len(pool))
        idxs = rng.choice(len(pool), size=k, replace=False)
        sampled.extend([pool[i] for i in idxs])

    # Shuffle the stratified results to remove any ordering bias from the bucket loop
    rng.shuffle(sampled)
    if len(sampled) < n:
        remaining = [r for r in records if r not in sampled]
        rng.shuffle(remaining)
        sampled.extend(remaining[: n - len(sampled)])

    return sampled[:n]

# Reporting

def build_report(tagging_results: List[TaggingResult],eval_results: List[EvaluationResult],output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_by_id = {e.movie_id: e for e in eval_results}

    rows = []
    for tr in tagging_results:
        ev = eval_by_id.get(tr.movie_id)
        row = {
            "movieId": tr.movie_id,
            "title": tr.title,
            "predicted_tags": ", ".join(tr.predicted_tags),
            "ground_truth_tags": ", ".join(tr.ground_truth_tags),
            "latency_ms": round(tr.latency_ms, 1),
            "tag_confidences": json.dumps({k: round(v, 2) for k, v in tr.confidences.items()}),
        }
        if ev:
            row.update({
                "jaccard": ev.jaccard,
                "precision": ev.precision,
                "recall": ev.recall,
                "f1": ev.f1,
                "novelty_rate": ev.novelty_rate,
                "exact_ground_truth_coverage": ev.exact_ground_truth_coverage,
                "sem_avg_max_cosine": ev.sem_avg_max_cosine,
                "sem_set_cosine": ev.sem_set_cosine,
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    tags_file = output_dir / "generated_movie_tags.csv"
    df.to_csv(tags_file, index=False)

    agg: Dict = {
        "total_movies_evaluated": len(df),
        "avg_predicted_tags_per_movie": round(
            df["predicted_tags"].apply(lambda x: len([t for t in x.split(",") if t.strip()])).mean(), 2
        ),
        "avg_latency_ms": round(df["latency_ms"].mean(), 1),
    }
    for col in ["jaccard", "precision", "recall", "f1", "novelty_rate",
                "exact_ground_truth_coverage", "sem_avg_max_cosine", "sem_set_cosine"]:
        if col in df.columns and df[col].notna().any():
            agg[f"avg_{col}"] = round(df[col].mean(), 4)

    metrics_file = output_dir / "evaluation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(agg, f, indent=4)

    for k, v in agg.items():
        print(f"{k:<35} {v}")
    
# Pipeline

def run_pipeline( imdb_path: Path,links_path: Path, tags_path: Path, movies_path: Path,
    prompt_path: Path, output_dir: Path, sample_size: int = 20, model_name: str = "llama-3.1-8b-instant") -> None:

    system_prompt = load_prompt(prompt_path)

    plots = load_imdb_plots(imdb_path, links_path)
    ground_truth_tags = load_ground_truth_tags(tags_path)
    meta = load_movie_metadata(movies_path)
    print(f"{len(plots):,} plots  |  {len(ground_truth_tags):,} movies with ground_truth tags")

    records: List[MovieRecord] = []
    for mid, plot in plots.items():
        if not plot or plot == "nan":
            continue
        ground_truth = ground_truth_tags.get(mid, [])
        if not ground_truth:
            continue
        title, genres = meta.get(mid, ("Unknown", []))
        records.append(MovieRecord(mid, title, plot, genres, ground_truth))

    print(f"{len(records):,} movies with plot + ground_truth tags")

    sample = stratified_sample(records, n=sample_size)

    tagger = MovieTaggerLLM(system_prompt=system_prompt, model_name=model_name)
    evaluator = TagEvaluator()
 
    tagging_results: List[TaggingResult] = []
    eval_results: List[EvaluationResult] = []

    for i, rec in enumerate(sample, 1):
        print(f"{i} out of {len(sample)} movieId={rec.movie_id}  '{rec.title[:50]}'")
        tags, confs, raw, latency = tagger.generate_tags(rec.plot)
        print(f"predicted: {tags}")
        print(f"ground_truth sample: {rec.ground_truth_tags[:5]}")

        tagging_results.append(TaggingResult(movie_id=rec.movie_id,title=rec.title,predicted_tags=tags,
            confidences=confs, ground_truth_tags=rec.ground_truth_tags, raw_llm_output=raw, latency_ms=latency))

        ev = evaluator.evaluate(rec.movie_id, tags, rec.ground_truth_tags)
        eval_results.append(ev)
        print(f"F1={ev.f1:.2f},  Jaccard={ev.jaccard:.2f},  Novelty={ev.novelty_rate:.2f}")

    build_report(tagging_results, eval_results, output_dir)



if __name__ == "__main__":
    BASE = Path("./data/movie_lens")
    IMDB_DIR = Path("./data/imdb")
    OUT_DIR = Path("./data/output_llm")
    PROMPT = Path("./prompt.txt")

    run_pipeline(
        imdb_path=IMDB_DIR / "imdb_data.parquet",
        links_path=BASE / "links.csv",
        tags_path=BASE / "tags.csv",
        movies_path=BASE / "movies.csv",
        prompt_path=PROMPT,
        output_dir=OUT_DIR,
        sample_size=20,
        model_name="llama-3.1-8b-instant",
    )
