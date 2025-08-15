from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient, UpdateOne
import numpy as np
import gc
import torch
from tqdm import tqdm
import time
import logging
import heapq
import numpy as np
from numpy.linalg import norm
from itertools import count

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = MongoClient("mongodb://localhost:27017/")
db = client["local"]
collection = db["optionProducts"]

schema = {
  "collection": "optionProducts",
  "fields": {
    "_id": "objectId",
    "audit": "object",
    "isActive": "boolean",
    "legacyOptionId": "int",
    "legacyProductId": "int",
    "option": "object",
    "optionActive": "boolean",
    "optionId": "objectId",
    "option_description": "string",
    "option_name": "string",
    "product": "object",
    "productActive": "boolean",
    "productId": "objectId",
    "products": "object",
    "searchKeywords": "array<string>",
    "searchText": "string"
  }
}


def getEmbeddingFields(document):
    products = document.get("products", {})

    fields = [
        products.get("product_name", ""),
        products.get("product_description", ""),
        products.get("product_brand_name", ""),
        products.get("product_style_name", ""),
        products.get("product_category_name", ""),
        products.get("product_category_group_name", "")
    ]
    return " ".join([f for f in fields if f])

def getEmbeddings(text):
    model = initializeModel()

    return model.encode(text).tolist()

#heap-based optimized local vector search
def optimizedLocalVectorHeapSearch(query_embedding, top_k=5, batch_size=1000, normalize=True):
    """
    Unique top-K by product (recommended):
    - Streams Mongo in batches
    - L2-normalizes -> cosine == dot
    - Keeps only top-K unique products using a min-heap (+ lazy stale entry cleanup)
    - Returns [{product_id, score, product, option}, ...] sorted by score desc
    """
    import numpy as np, heapq
    from numpy.linalg import norm

    def get_product_id(doc):
        p = doc.get("product")
        if isinstance(p, dict):
            for k in ("id", "productId", "legacyProductId", "_id"):
                v = p.get(k)
                if v is not None and v != "":
                    return str(v)
        for k in ("productId", "legacyProductId"):
            v = doc.get(k)
            if v is not None and v != "":
                return str(v)
        return None

    # Prepare query vector
    q = np.asarray(query_embedding, dtype=np.float32)
    if normalize:
        q = q / (norm(q) + 1e-9)
    q = q.reshape(1, -1)

    # Mongo cursor (only needed fields)
    cursor = collection.find(
        {"embedding": {"$exists": True}},
        {"_id": 1, "embedding": 1, "option": 1, "product": 1, "productId": 1, "legacyProductId": 1}
    ).batch_size(batch_size)

    # Min-heap of (score, tie, pid); entry_finder tracks the *current* best per pid
    heap = []
    entry_finder = {}   # pid -> (score, tie)
    doc_by_pid = {}     # pid -> best doc

    def clean_top():
        """Pop stale heap entries until top matches entry_finder."""
        while heap:
            s, tie, pid = heap[0]
            cur = entry_finder.get(pid)
            if cur is None or cur[0] != s or cur[1] != tie:
                heapq.heappop(heap)  # stale
            else:
                return s, tie, pid
        return None

    def consider(pid, score, doc):
        """Update structures with a candidate (pid, score, doc)."""
        # Tiebreaker: string(pid) so earlier ObjectId wins ties; swap to '-createdAt' etc. if desired
        tie = str(pid)
        if pid in entry_finder:
            cur_s, cur_tie = entry_finder[pid]
            if score > cur_s or (score == cur_s and tie < cur_tie):
                entry_finder[pid] = (score, tie)
                doc_by_pid[pid] = doc
                heapq.heappush(heap, (score, tie, pid))  # old entry becomes stale
        else:
            if len(entry_finder) < top_k:
                entry_finder[pid] = (score, tie)
                doc_by_pid[pid] = doc
                heapq.heappush(heap, (score, tie, pid))
            else:
                top_min = clean_top()
                if top_min is None:
                    entry_finder[pid] = (score, tie)
                    doc_by_pid[pid] = doc
                    heapq.heappush(heap, (score, tie, pid))
                else:
                    min_score, min_tie, min_pid = top_min
                    if score > min_score or (score == min_score and tie < min_tie):
                        heapq.heappop(heap)               # evict current min
                        entry_finder.pop(min_pid, None)
                        doc_by_pid.pop(min_pid, None)
                        entry_finder[pid] = (score, tie)
                        doc_by_pid[pid] = doc
                        heapq.heappush(heap, (score, tie, pid))

    # Batch buffers
    docs, embeds = [], []

    def flush_batch():
        if not embeds:
            return
        M = np.vstack(embeds).astype(np.float32, copy=False)  # (N, d)
        if normalize:
            M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        scores = (q @ M.T).ravel()  # (N,)
        for s, d in zip(scores, docs):
            pid = get_product_id(d)
            if pid is not None:
                consider(pid, float(s), d)
        docs.clear(); embeds.clear()

    for doc in cursor:
        emb = doc.get("embedding")
        if emb is None:
            continue
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 1 or emb.shape[0] != q.shape[1]:
            continue
        docs.append(doc); embeds.append(emb)
        if len(embeds) >= batch_size:
            flush_batch()
    flush_batch()

    # Extract winners (desc by score, then by tie)
    winners = []
    while True:
        top_min = clean_top()
        if top_min is None:
            break
        s, tie, pid = heapq.heappop(heap)
        # Ensure consistency; clean_top already guarantees it
        winners.append((s, tie, pid))
        entry_finder.pop(pid, None)

    winners.sort(key=lambda t: (-t[0], t[1]))
    out = []
    for s, tie, pid in winners:
        d = doc_by_pid[pid]
        out.append({
            "product_id": pid,
            "score": round(float(s), 6),
            "product": d.get("product"),
            "option": d.get("option"),  # option that yielded the best score
        })
    return out


#has cosine similarity optimization
def optimizedLocalVectorCosineSearch(query_embedding, top_k=10, batch_size=1000):
    """Return top-K UNIQUE products with their best (highest) score."""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    def get_product_id(doc):
        p = doc.get("product")
        if isinstance(p, dict):
            for k in ("id", "productId", "legacyProductId", "_id"):
                if p.get(k) is not None:
                    return str(p[k])
        for k in ("productId", "legacyProductId"):
            if doc.get(k) is not None:
                return str(doc[k])
        return None

    query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    best_by_pid = {}  # pid -> (score, doc)

    cursor = collection.find(
        {"embedding": {"$exists": True}},
        {"_id": 1, "embedding": 1, "option": 1, "product": 1, "productId": 1, "legacyProductId": 1}
    ).batch_size(batch_size)

    batch_docs, batch_embeddings = [], []

    def flush_batch():
        nonlocal batch_docs, batch_embeddings, best_by_pid
        if not batch_embeddings:
            return
        M = np.asarray(batch_embeddings, dtype=np.float32)
        sims = cosine_similarity(query_vec, M).ravel()  # (N,)
        for s, d in zip(sims, batch_docs):
            pid = get_product_id(d)
            if not pid:
                continue
            s = float(s)
            prev = best_by_pid.get(pid)
            if prev is None or s > prev[0]:
                best_by_pid[pid] = (s, d)
        batch_docs.clear()
        batch_embeddings.clear()

    for doc in cursor:
        emb = doc.get("embedding")
        if emb is None:
            continue
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 1 or emb.shape[0] != query_vec.shape[1]:
            continue
        batch_docs.append(doc)
        batch_embeddings.append(emb)
        if len(batch_embeddings) >= batch_size:
            flush_batch()

    flush_batch()

    # Sort products by their BEST score (desc), take top_k, and include score in output
    winners = sorted(best_by_pid.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]
    return [
        {
            "product_id": pid,
            "score": round(score, 6),
            "product": doc.get("product"),
            "option": doc.get("option"),  # the option that produced the best score
        }
        for pid, (score, doc) in winners
    ]




# def localVectorSearch(query_embedding, top_k=5):
#     query_vec = np.array(query_embedding).reshape(1, -1)
#     results = []

#     for doc in collection.find({"embedding": {"$exists": True}}):
#         doc_vec = np.array(doc["embedding"]).reshape(1, -1)
#         score = cosine_similarity(query_vec, doc_vec)[0][0]
#         results.append((score, doc))

#     top_results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]

#     print("\n=== Top Matching Documents (Local Cosine Search) ===")
#     for score, doc in top_results:
#         print(f"Score: {score:.4f}")
#         print(f"Document ID: {doc['_id']}")
#         print("Option:")
#         for k, v in doc.get("option", {}).items():
#             print(f"  {k}: {v}")
#         print("Product:")
#         for k, v in doc.get("product", {}).items():
#             print(f"  {k}: {v}")
#         print("-" * 40)

#     return [doc for _, doc in top_results]

@lru_cache(maxsize=1)
def initializeModel():
    """Initialize the sentence transformer model with optimizations"""
    model = SentenceTransformer("intfloat/e5-base-v2", device="mps")
    
    # Enable mixed precision for faster inference (if using CUDA)
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        model.to(torch.device("mps"))
    else:
        model.to(torch.device("cpu"))

    logger.info(f"Model loaded on device: {model.device}")
    return model

def processBatchEmbeddings(texts, model, batch_size=128):
    """Generate embeddings for a batch of texts with optimal settings"""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_tensor=False,  # Keep as numpy for memory efficiency
        normalize_embeddings=True  # Normalize for better similarity computation
    )
    return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

# def addEmbeddingFieldOptimized(chunk_size=2000, embedding_batch_size=64):
#     """
#     Optimized function to add embedding fields to documents
    
#     Args:
#         chunk_size: Number of documents to process in each chunk
#         embedding_batch_size: Batch size for embedding generation
#         skip_existing: Skip documents that already have embeddings
#     """
    
#     model = initializeModel()
    
#     # Get total count for progress tracking
#     # filter_query = {"embedding": {"$exists": False}} if skip_existing else {}
#     docs_to_process = collection.count_documents({})
#     logger.info(f"Documents needing embedding: {docs_to_process}")
    
#     if docs_to_process == 0:
#         logger.info("All documents already have embeddings!")
#         return
    
#     processed_count = 0
    
#     # Process in chunks
#     with tqdm(total=docs_to_process, desc="Processing embeddings") as pbar:
#         # skip = 0
        
#         while True:
#         #    try:
#                 # Fetch chunk of documents
#             cursor = collection.find().limit(chunk_size)
#             documents = list(cursor)
            
#             if not documents:
#                 break
            
#             # Prepare data for batch processing
#             doc_ids = []
#             texts = []
            
#             for doc in documents:
#                 embedding_text = getEmbeddingFields(doc)
#                 if embedding_text.strip():  # Only process non-empty texts
#                     doc_ids.append(doc["_id"])
#                     texts.append(embedding_text)
            
#             if not documents:
#                 break
#                 # skip += len(documents)
#                 # continue
                
            
#             logger.info(f"Processing chunk: {len(texts)} documents")
            
#             # Generate embeddings in batches
#             embeddings = processBatchEmbeddings(texts, model, embedding_batch_size)
            
#             # Prepare bulk update operations
#             bulk_operations = []
#             for doc_id, embedding in zip(doc_ids, embeddings):
#                 bulk_operations.append(
#                     UpdateOne(
#                         {"_id": doc_id},
#                         {"$set": {"embedding": embedding}},
#                         upsert=True
#                     )
#                 )
            
#             # Execute bulk write
#             if bulk_operations:
#                 result = collection.bulk_write(bulk_operations, ordered=False)
#                 logger.info(f"Updated {result.modified_count} documents")
#                 processed_count += result.modified_count
            
#             # Clear memory
#             del embeddings, bulk_operations, documents, texts
#             gc.collect()
            
#             # Update progress
#             pbar.update(len(doc_ids))
#             # skip += chunk_size

#             if pbar.total:  # guard if total=None
#                 pbar.set_postfix(percent=f"{(pbar.n / pbar.total):.1%}")
#             else:
#                 pbar.set_postfix(percent="n/a")
            
#             # Add small delay to prevent overwhelming the system
#             time.sleep(0.1)
                
#             # except Exception as e:
#             #     logger.error(f"Error processing chunk at skip {skip}: {e}")
#             #     skip += chunk_size
#             #     continue
    
#     logger.info(f"Completed! Processed {processed_count} documents total")

def addEmbeddingFieldOptimized(chunk_size=2000, embedding_batch_size=64):
    """
    Recompute embeddings for ALL documents (overwrite existing).
    Uses _id pagination (no skip), bulk writes, and tqdm % postfix.
    """

    model = initializeModel()

    docs_to_process = collection.count_documents({})
    logger.info(f"Documents to (re)embed: {docs_to_process}")
    if docs_to_process == 0:
        logger.info("No documents found.")
        return

    processed_count = 0
    last_id = None

    with tqdm(total=docs_to_process, desc="Processing embeddings", unit="doc", dynamic_ncols=True) as pbar:
        while True:
            # Page by _id to avoid fetching the same first page forever
            query = {"_id": {"$gt": last_id}} if last_id is not None else {}
            cursor = (collection.find(query)
                                .sort("_id", 1)
                                .limit(chunk_size)
                                .batch_size(chunk_size))
            documents = list(cursor)
            if not documents:
                break

            last_id = documents[-1]["_id"]

            # Build batch for embedding
            doc_ids, texts = [], []
            for doc in documents:
                embedding_text = getEmbeddingFields(doc)
                if embedding_text and embedding_text.strip():
                    doc_ids.append(doc["_id"])
                    texts.append(embedding_text)

            logger.info(f"Processing chunk: {len(documents)} docs | to-embed(non-empty): {len(texts)}")

            # Generate embeddings and write back (overwrite)
            bulk_operations = []
            if texts:
                embeddings = processBatchEmbeddings(texts, model, embedding_batch_size)
                for doc_id, embedding in zip(doc_ids, embeddings):
                    bulk_operations.append(
                        UpdateOne(
                            {"_id": doc_id},
                            {"$set": {"embedding": embedding}},  # overwrite existing
                            upsert=False
                        )
                    )

            if bulk_operations:
                result = collection.bulk_write(bulk_operations, ordered=False)
                logger.info(f"Modified {result.modified_count} docs in this chunk")

            # Progress & percent (Option B)
            docs_this_batch = len(documents)
            processed_count += docs_this_batch

            # Progress & percent (Option B)
            try:
                pbar.update(docs_this_batch)
            except Exception:
                # fallback in case the terminal got funky
                pbar.n = min(pbar.n + docs_this_batch, pbar.total or pbar.n + docs_this_batch)
                pbar.refresh()

            if pbar.total:
                pbar.set_postfix(percent=f"{(pbar.n / pbar.total):.1%}")
            else:
                pbar.set_postfix(percent="n/a")

            # Free memory
            del documents, texts, doc_ids
            gc.collect()

            time.sleep(0.1)

    logger.info(f"Completed! Processed {processed_count} documents total")


# Optional: use the shared model everywhere to avoid reloading
# def getEmbeddings(text):
#     m = get_model()
#     emb = m.encode([text], batch_size=1, show_progress_bar=False,
#                    convert_to_tensor=False, normalize_embeddings=True)
#     return emb[0]


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

def main():
    #addEmbeddingField()
    CHUNK_SIZE = 2000          # Process 2000 documents at a time
    EMBEDDING_BATCH_SIZE = 64   # Generate 64 embeddings per batch
    SKIP_EXISTING = True        # Skip documents that already have embeddings
    
    logger.info("Starting optimized embedding generation...")
    
    # Add embeddings to all documents
    # addEmbeddingFieldOptimized(
    #     chunk_size=CHUNK_SIZE,
    #     embedding_batch_size=EMBEDDING_BATCH_SIZE,
    # )
    
    logger.info("Embedding generation completed successfully!")

    while True:
        userQuery = input("Enter a query for Option Search:")
        if userQuery.lower() == 'quit':
            break

        query_embedding = getEmbeddings(userQuery)
        #topK_results = optimizedLocalVectorCosineSearch(query_embedding) #26.18 secs
        topK_results = optimizedLocalVectorHeapSearch(query_embedding, top_k=10) #17.28 secs
        print(f"\nTop {len(topK_results)} results for query: '{userQuery}'")
        for idx, doc in enumerate(topK_results, 1):
            print(f"\nResult {idx}:")
            print(f"Score: {doc['score']}")
            print("Product Details:")
            for k, v in doc.get("product", {}).items():
                print(f"  {k}: {v}")
            print("Option Details:")
            for k, v in doc.get("option", {}).items():
                print(f"  {k}: {v}")
            print("-" * 40)

if __name__ == "__main__":
    main()
else:
    print("This script is intended to be run as a standalone program.")
