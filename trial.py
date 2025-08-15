#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient, UpdateOne
import numpy as np
import torch
from tqdm import tqdm
import logging
import gc
import time

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("option_vectorizer")

# ----------------------------
# MongoDB
# ----------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["local"]
collection = db["optionProducts"]

# ----------------------------
# Embedding config & model cache
# ----------------------------
EMB_MODEL = "intfloat/e5-base-v2"
EMB_DIM = 768

_MODEL = None
def get_model():
    """Load the model once (cached)."""
    global _MODEL
    if _MODEL is None:
        # Prefer CUDA if available; else Apple MPS; else CPU
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        _MODEL = SentenceTransformer(EMB_MODEL, device=device)
        if device == "cuda":
            _MODEL.half()  # FP16 on CUDA
        logger.info(f"Model {EMB_MODEL} loaded on device: {device}")
    return _MODEL

def embed_texts(texts, batch_size=64, normalize=True):
    m = get_model()
    embs = m.encode(texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=False,
                    normalize_embeddings=normalize)
    # returns list[list[float]]
    return embs.tolist() if isinstance(embs, np.ndarray) else embs

def getEmbeddings(text):
    """Convenience for single string."""
    return embed_texts([text], batch_size=1, normalize=True)[0]

# ----------------------------
# Text extraction for embeddings
# ----------------------------
def doc_text_for_embedding(doc):
    """
    Build a single string from doc fields to embed.
    Prefers 'products' fields, falls back to top-level search text/keywords/option fields.
    """
    products = doc.get("products") or {}
    fields = [
        products.get("product_name", ""),
        products.get("product_description", ""),
        products.get("product_brand_name", ""),
        products.get("product_style_name", ""),
        products.get("product_category_name", ""),
        products.get("product_category_group_name", ""),
        doc.get("option_name", ""),
        doc.get("option_description", ""),
        doc.get("searchText", ""),
        " ".join(doc.get("searchKeywords", []) or []),
    ]
    return " ".join(f for f in fields if f).strip()

# ----------------------------
# Health / visibility
# ----------------------------
def count_targets(base_query=None):
    base_query = base_query or {}
    total = collection.count_documents({})
    with_emb = collection.count_documents({"embedding": {"$exists": True}})
    wrong_dim = collection.count_documents({
        "embedding": {"$exists": True},
        "$expr": {"$ne": [{"$size": "$embedding"}, EMB_DIM]}
    })
    missing = total - with_emb
    logger.info(f"Docs total={total}, with_embedding={with_emb}, wrong_dim={wrong_dim}, missing={missing}")
    if base_query:
        target_count = collection.count_documents(base_query)
        logger.info(f"Targets matching query: {target_count}")
    return total, with_emb, wrong_dim, missing

def report_by_model_dim():
    agg = list(collection.aggregate([
        {"$match": {"embedding": {"$exists": True}}},
        {"$project": {
            "embedding_model": 1,
            "dim": {"$cond": [{"$isArray": "$embedding"}, {"$size": "$embedding"}, -1]}
        }},
        {"$group": {"_id": {"model": "$embedding_model", "dim": "$dim"}, "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]))
    logger.info(f"By model/dim: {agg}")

# ----------------------------
# Embedding writer (paginated, progress-aware)
# ----------------------------
def addEmbeddingFieldOptimized(chunk_size=2000, embedding_batch_size=64, only_missing_or_wrong=True, sleep_between=0.0):
    """
    Rebuild embeddings with stable _id pagination and explicit progress logs.
    - only_missing_or_wrong=True: process docs missing embeddings or wrong dimension
    """
    get_model()  # ensure single load

    base_query = {}
    if only_missing_or_wrong:
        base_query = {"$or": [
            {"embedding": {"$exists": False}},
            {"$expr": {"$ne": [{"$size": "$embedding"}, EMB_DIM]}}
        ]}

    total_targets = collection.count_documents(base_query)
    logger.info(f"Will embed/re-embed {total_targets} documents (only_missing_or_wrong={only_missing_or_wrong})")

    last_id = None
    processed = 0
    skipped_empty_text = 0

    with tqdm(total=total_targets, desc="Embedding docs", unit="doc") as pbar:
        while True:
            query = dict(base_query)
            if last_id:
                query["_id"] = {"$gt": last_id}

            batch = list(collection.find(query).sort("_id", 1).limit(chunk_size))
            if not batch:
                break

            ids, texts = [], []
            for d in batch:
                t = doc_text_for_embedding(d)
                if t:
                    ids.append(d["_id"])
                    texts.append(t)
                else:
                    skipped_empty_text += 1

            if ids:
                embs = embed_texts(texts, batch_size=embedding_batch_size, normalize=True)
                ops = [
                    UpdateOne(
                        {"_id": _id},
                        {"$set": {
                            "embedding": emb,
                            "embedding_model": EMB_MODEL,
                            "embedding_dim": EMB_DIM
                        }},
                        upsert=False
                    )
                    for _id, emb in zip(ids, embs)
                ]
                res = collection.bulk_write(ops, ordered=False)
                processed += len(ids)
                pbar.update(len(ids))
                tqdm.write(f"Chunk wrote {len(ids)} | total processed {processed}")

                # free memory
                del embs, ops
                gc.collect()

            last_id = batch[-1]["_id"]

            if sleep_between > 0:
                time.sleep(sleep_between)

    logger.info(f"Embedding pass complete. processed={processed}, skipped_empty_text={skipped_empty_text}")

# ----------------------------
# Local vector search (robust)
# ----------------------------
def localVectorSearch(query_embedding, top_k=5):
    """
    Cosine search over locally stored embeddings.
    Filters by expected model & dimension; skips any stragglers defensively.
    """
    query_vec = np.array(query_embedding).reshape(1, -1)
    results = []
    bad_dim = 0

    cursor = collection.find({
        "embedding": {"$exists": True},
        "embedding_dim": EMB_DIM,
        "embedding_model": EMB_MODEL
    }, projection=["embedding", "option", "product"])

    for doc in cursor:
        emb = doc.get("embedding")
        if not emb or len(emb) != EMB_DIM:
            bad_dim += 1
            continue
        doc_vec = np.array(emb).reshape(1, -1)
        score = cosine_similarity(query_vec, doc_vec)[0][0]
        results.append((score, doc))

    if bad_dim:
        logger.warning(f"Skipped {bad_dim} docs with incorrect dim")

    top_results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]

    print("\n=== Top Matching Documents (Local Cosine Search) ===")
    for score, doc in top_results:
        print(f"Score: {score:.4f}")
        print(f"Document ID: {doc['_id']}")
        print("Option:")
        for k, v in (doc.get("option") or {}).items():
            print(f"  {k}: {v}")
        print("Product:")
        for k, v in (doc.get("product") or {}).items():
            print(f"  {k}: {v}")
        print("-" * 40)

    return [doc for _, doc in top_results]

# ----------------------------
# Optional: hard reset
# ----------------------------
def wipe_all_embeddings():
    res = collection.update_many({"embedding": {"$exists": True}},
                                 {"$unset": {"embedding": "", "embedding_dim": "", "embedding_model": ""}})
    logger.info(f"Wiped embedding fields from {res.modified_count} docs")

# ----------------------------
# (Optional) simple IR metrics helpers
# ----------------------------
def compute_precision_at_k(retrieved_ids, relevant_ids, k):
    retrieved_top_k = retrieved_ids[:k]
    num_relevant = len(set(retrieved_top_k) & set(relevant_ids))
    return num_relevant / k

def compute_recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved_top_k = retrieved_ids[:k]
    num_relevant = len(set(retrieved_top_k) & set(relevant_ids))
    return num_relevant / len(relevant_ids) if relevant_ids else 0

def compute_f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# ----------------------------
# Main
# ----------------------------
def main():
    logger.info("=== Pre-run health ===")
    count_targets()
    report_by_model_dim()

    CHUNK_SIZE = 2000
    EMBEDDING_BATCH_SIZE = 64

    logger.info("Starting embedding (missing or wrong dim only)…")
    addEmbeddingFieldOptimized(
        chunk_size=CHUNK_SIZE,
        embedding_batch_size=EMBEDDING_BATCH_SIZE,
        only_missing_or_wrong=True,
        sleep_between=0.0  # increase if you want to throttle
    )

    logger.info("=== Post-run health ===")
    count_targets()
    report_by_model_dim()

    while True:
        try:
            userQuery = input("Enter a query for Option Search (or 'quit'): ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not userQuery or userQuery.strip().lower() == 'quit':
            break
        query_embedding = getEmbeddings(userQuery)
        _ = localVectorSearch(query_embedding)

if __name__ == "__main__":
    main()
else:
    print("This script is intended to be run as a standalone program.")
