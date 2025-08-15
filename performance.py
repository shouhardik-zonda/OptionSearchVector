# from sentence_transformers import SentenceTransformer
# from pymongo import MongoClient
from unittest import skip
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient, UpdateOne
import numpy as np
import gc
import torch
from tqdm import tqdm
import time
import logging
import json
import sys
import random

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
    #model = SentenceTransformer("all-MiniLM-L6-v2")
    #model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
    #model = SentenceTransformer("Salesforce/SFR-Embedding-2_R", device="mps")
    # model = SentenceTransformer("intfloat/e5-base-v2", device = "mps")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device="mps")

    return model.encode(text).tolist()

def addEmbeddingField():
    for document in collection.find():
        embedding_fields = getEmbeddingFields(document)
        if embedding_fields:
            embedding = getEmbeddings(embedding_fields)
            collection.update_one(
                {"_id": document["_id"]},
                {"$set": {"embedding": embedding}},
                upsert=True
            )

def localVectorSearch(query_embedding, top_k=10):
    query_vec = np.array(query_embedding).reshape(1, -1)
    results = []

    for doc in collection.find({"embedding": {"$exists": True}}):
        doc_vec = np.array(doc["embedding"]).reshape(1, -1)
        score = cosine_similarity(query_vec, doc_vec)[0][0]
        results.append((score, doc))

    # Step 2: Sort all results by descending score
    sorted_results = sorted(results, key=lambda x: x[0], reverse=True)

    # Step 3: Filter by unique (productId, optionId)
    seen = set()
    top_unique_results = []
    for score, doc in sorted_results:
        product_id = str(doc.get("productId", ""))
        key = (product_id)
        
        if key not in seen:
            seen.add(key)
            top_unique_results.append((score, doc))
        
        if len(top_unique_results) >= top_k:
            break

    # Step 4: Print results
    print("\n=== Top Unique Matching Documents (Local Cosine Search) ===")
    for score, doc in top_unique_results:
        print(f"Score: {score:.4f}")
        print(f"Document ID: {doc['_id']}")
        print("Option:")
        for k, v in doc.get("option", {}).items():
            print(f"  {k}: {v}")
        print("Product:")
        for k, v in doc.get("product", {}).items():
            print(f"  {k}: {v}")
        print("-" * 40)

    return top_unique_results

def initializeModel():
    """Initialize the sentence transformer model with optimizations"""
    #model = SentenceTransformer("Salesforce/SFR-Embedding-2_R", device="mps")
    #model = SentenceTransformer("intfloat/e5-base-v2", device="mps")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device="mps")

    # Enable mixed precision for faster inference (if using CUDA)
    if torch.cuda.is_available():
        model.half()  # Use FP16 for faster processing
        
    logger.info(f"Model loaded on device: {model.device}")
    return model

def processBatchEmbeddings(texts, model, batch_size=64):
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
#         skip = 0
        
#         while True:
#         #    try:
#                 # Fetch chunk of documents
#             #cursor = collection.find().limit(chunk_size)
#             cursor = collection.find().skip(skip).limit(chunk_size)
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
            
#             if not texts:
#                 skip += len(documents)  # ✅ Still need to advance skip
#                 pbar.update(len(documents))
#                 continue
                
            
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
#                 logger.info(f"Matched: {result.matched_count}, Modified: {result.modified_count}, Upserts: {len(result.upserted_ids)}")
#                 processed_count += result.modified_count
            
#             # Clear memory
#             del embeddings, bulk_operations, documents, texts
#             gc.collect()
            
#             # Update progress
#             pbar.update(len(doc_ids))
#             # skip += chunk_size
            
#             # Add small delay to prevent overwhelming the system
#             time.sleep(0.1)
                
#             # except Exception as e:
#             #     logger.error(f"Error processing chunk at skip {skip}: {e}")
#             #     skip += chunk_size
#             #     continue
    
#     logger.info(f"Completed! Processed {processed_count} documents total")

def addEmbeddingFieldOptimized(chunk_size=2000, embedding_batch_size=64):
    """Optimized function to add/update embedding fields for all documents"""
    model = initializeModel()
    
    # Get total count for progress tracking
    total_docs = collection.count_documents({})
    logger.info(f"Total documents to process: {total_docs}")
    
    if total_docs == 0:
        logger.info("No documents found!")
        return
    
    processed_count = 0
    skip = 0
    
    with tqdm(total=total_docs, desc="Processing embeddings") as pbar:
        while skip < total_docs:
            try:
                # Fetch chunk of documents
                cursor = collection.find().skip(skip).limit(chunk_size)
                documents = list(cursor)
                
                if not documents:
                    break
                
                # Prepare data for batch processing
                doc_ids = []
                texts = []
                
                for doc in documents:
                    embedding_text = getEmbeddingFields(doc)
                    if embedding_text.strip():  # Only process non-empty texts
                        doc_ids.append(doc["_id"])
                        texts.append(embedding_text)
                
                logger.info(f"Processing chunk: {len(texts)} documents")
                
                if texts:
                    # Generate embeddings in batches
                    embeddings = processBatchEmbeddings(texts, model, embedding_batch_size)
                    
                    # Prepare bulk update operations
                    bulk_operations = [
                        UpdateOne(
                            {"_id": doc_id},
                            {"$set": {"embedding": embedding}},
                        )
                        for doc_id, embedding in zip(doc_ids, embeddings)
                    ]
                    
                    # Execute bulk write
                    result = collection.bulk_write(bulk_operations, ordered=False)
                    chunk_processed = len(bulk_operations)
                    processed_count += len(bulk_operations)
                    logger.info(
                        f"Chunk update results - "
                        f"Total ops: {chunk_processed}, "
                        f"Matched: {result.matched_count}, "
                        f"Modified: {result.modified_count}"
                    )
                
                # Update progress and skip
                skip += len(documents)
                pbar.update(len(documents))
                
                # Clear memory
                del embeddings, bulk_operations, documents, texts
                gc.collect()
                
                # Add small delay
                time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error processing chunk at skip {skip}: {e}")
                skip += chunk_size
                continue
    
    logger.info(f"Completed! Processed {processed_count} documents total")

def create_test_set(num_queries=100, samples_per_query=20):
    """Generate a labeled test set from your data"""
    test_set = {}
    
    # Option 2: Generate synthetic queries from product data
    queries = generate_synthetic_queries(num_queries)
    for query in tqdm(queries, desc="Building test set"):
        print(f"Processing query: {query}")
        query_embedding = getEmbeddings(query)
        results = localVectorSearch(query_embedding, top_k=samples_per_query*3)
        
        # Automatically label based on similarity score
        labeled_results = []
        #query_vec = np.array(query_embedding).reshape(-1,1)
        for score, doc in results[:samples_per_query]:
            doc_embedding = doc.get("embedding")
            #score = cosine_similarity(query_vec, doc_embedding)
            if score > 0.85:
                label = 3  # Perfect match
            elif score > 0.75:
                label = 2  # Good match
            elif score > 0.5:
                label = 1  # Partial match
            else:
                label = 0  # Irrelevant
            
            labeled_results.append({
                "product_id": doc["_id"],
                "score": score,
                "label": label,
                "product_name": doc.get("product", {}).get("product_name", "")
            })
        
        test_set[query] = {
            "query_embedding": query_embedding,
            "results": labeled_results
        }
    
    return test_set

# def generate_synthetic_queries(num_queries):
#     """Generate realistic queries from product data"""
#     queries = set()
    
#     # Sample from different product fields
#     for doc in collection.aggregate([{"$sample": {"size": num_queries*2}}]):
#         product = doc.get("product", {})
        
#         # Create different query types
#         queries.add(product.get("product_name", ""))
#         queries.add(f"{product.get('product_brand_name', '')} {product.get('product_style_name', '')}")
#         queries.add(f"{product.get('product_category_name', '')} near me")
        
#         if len(queries) >= num_queries:
#             break
    
#     return list(queries)[:num_queries]

def generate_synthetic_queries(num_queries):
    """Fixed version that handles empty queries and ensures valid output"""
    
    # Query type counts
    num_category     = int(0.25 * num_queries)
    num_feature      = int(0.30 * num_queries)
    num_brand        = int(0.20 * num_queries)
    num_exact        = int(0.15 * num_queries)
    num_use_case     = num_queries - (num_category + num_feature + num_brand + num_exact)

    # Storage per category
    category_qs, feature_qs, brand_qs, exact_qs, usecase_qs = [], [], [], [], []
    
    # Increase sample size to ensure we get enough valid data
    sample_size = num_queries * 10
    cursor = collection.aggregate([{"$sample": {"size": sample_size}}])

    for doc in cursor:
        product = doc.get("product", {})
        name = product.get("product_name", "").strip()
        desc = product.get("product_description", "").strip()
        brand = product.get("product_brand_name", "").strip()
        style = product.get("product_style_name", "").strip()
        category = product.get("product_category_name", "").strip()
        category_group = product.get("product_category_group_name", "").strip()

        # Category queries - FIX: Check for empty strings and avoid duplicates
        if category and len(category_qs) < num_category:
            if category not in category_qs:
                category_qs.append(category)
        
        # Feature-based queries - FIX: Limit description length
        if style and desc and len(feature_qs) < num_feature:
            # Truncate long descriptions to create manageable queries
            desc_short = desc[:100] + "..." if len(desc) > 100 else desc
            feature_query = f"{style} {desc_short}".strip()
            if feature_query and feature_query not in feature_qs:
                feature_qs.append(feature_query)
        
        # Brand-specific queries
        if brand and category and len(brand_qs) < num_brand:
            brand_query = f"{brand} {category}".strip()
            if brand_query and brand_query not in brand_qs:
                brand_qs.append(brand_query)
        
        # Exact match
        if name and len(exact_qs) < num_exact:
            if name not in exact_qs:
                exact_qs.append(name)

        # Use-case queries (simplified)
        if category and len(usecase_qs) < num_use_case:
            usecase_query = f"{category.lower()} for kitchen"
            if usecase_query and usecase_query not in usecase_qs:
                usecase_qs.append(usecase_query)
        
        # Early exit if we have enough queries
        if (len(category_qs) >= num_category and
            len(feature_qs) >= num_feature and
            len(brand_qs) >= num_brand and
            len(exact_qs) >= num_exact and
            len(usecase_qs) >= num_use_case):
            break

    # Combine all queries
    all_queries = category_qs + feature_qs + brand_qs + exact_qs + usecase_qs
    
    # FIX: Filter out any empty or None queries
    valid_queries = [q for q in all_queries if q and q.strip()]
    
    # FIX: Add fallback queries if we don't have enough
    if len(valid_queries) < num_queries:
        fallback_queries = [
            "kitchen tile", "bathroom tile", "floor tile", "wall tile",
            "countertop", "modern tile", "ceramic tile", "stone tile",
            "subway tile", "marble tile", "wood flooring", "vinyl flooring"
        ]
        
        for fallback in fallback_queries:
            if fallback not in valid_queries:
                valid_queries.append(fallback)
                if len(valid_queries) >= num_queries:
                    break
    
    random.shuffle(valid_queries)
    return valid_queries[:num_queries]

def evaluate_search_performance(test_set, k_values=[1, 3, 5, 10]):
    """Evaluate performance across multiple metrics"""
    metrics = {
        f"P@{k}": [] for k in k_values
    }
    metrics.update({
        f"R@{k}": [] for k in k_values
    })
    metrics.update({
        "mAP": [],
        "mRR": [],
        "average_ndcg": []
    })
    
    for query, data in test_set.items():
        relevant_ids = [
            r["product_id"] for r in data["results"] if r["label"] >= 2
        ]
        retrieved_ids = [r["product_id"] for r in data["results"]]
        relevance_scores = [r["label"] for r in data["results"]]
        
        # Precision and Recall at different K values
        for k in k_values:
            metrics[f"P@{k}"].append(
                compute_precision_at_k(retrieved_ids, relevant_ids, k)
            )
            metrics[f"R@{k}"].append(
                compute_recall_at_k(retrieved_ids, relevant_ids, k)
            )
        
        # Mean Average Precision
        metrics["mAP"].append(
            compute_average_precision(retrieved_ids, relevant_ids)
        )
        
        # Mean Reciprocal Rank
        metrics["mRR"].append(
            compute_reciprocal_rank(retrieved_ids, relevant_ids)
        )
        
        # Normalized Discounted Cumulative Gain
        metrics["average_ndcg"].append(
            compute_ndcg(relevance_scores, retrieved_ids, relevant_ids)
        )
    
    # Aggregate results
    return {
        metric: np.mean(values) for metric, values in metrics.items()
    }

def compute_average_precision(retrieved_ids, relevant_ids):
    """Compute Average Precision for a single query"""
    precisions = []
    num_relevant = 0
    
    for i, pid in enumerate(retrieved_ids, 1):
        if pid in relevant_ids:
            num_relevant += 1
            precisions.append(num_relevant / i)
    
    return sum(precisions) / len(relevant_ids) if relevant_ids else 0

def compute_reciprocal_rank(retrieved_ids, relevant_ids):
    """Compute Reciprocal Rank for a single query"""
    for i, pid in enumerate(retrieved_ids, 1):
        if pid in relevant_ids:
            return 1 / i
    return 0

def compute_ndcg(relevance_scores, retrieved_ids, relevant_ids, k=10):
    """Compute NDCG for a single query"""
    # Ideal ranking (sorted by relevance)
    ideal = sorted(relevance_scores, reverse=True)[:k]
    
    # Actual ranking
    actual = []
    for pid in retrieved_ids[:k]:
        idx = retrieved_ids.index(pid)
        actual.append(relevance_scores[idx])
    
    return ndcg_score([ideal], [actual], k=k)

def load_human_query_test_set(filepath="human_queries.json"):
    with open(filepath, "r") as f:
        queries = json.load(f)

    test_set = {}
    for entry in tqdm(queries, desc="Processing human queries"):
        query_id = entry["query_id"]
        query_text = entry["query_text"]
        labels = entry["relevance_labels"]

        query_embedding = getEmbeddings(query_text)
        results = localVectorSearch(query_embedding, top_k=50)

        labeled_results = []
        for score, doc in results:
            doc_id = str(doc["_id"])
            label = labels.get(doc_id, 0)
            labeled_results.append({
                "product_id": doc_id,
                "score": score,
                "label": label,
                "product_name": doc.get("product", {}).get("product_name", "")
            })

        test_set[query_id] = {
            "query_text": query_text,
            "query_embedding": query_embedding,
            "results": labeled_results
        }

    return test_set

def evaluate_from_human_json(filepath="human_queries.json", top_k=50, k_values=[1, 3, 5, 10]):
    with open(filepath, "r") as f:
        queries = json.load(f)

    test_set = {}
    for entry in tqdm(queries, desc="Evaluating human queries"):
        query_id = entry["query_id"]
        query_text = entry["query_text"]
        relevance_dict = entry["relevance_labels"]

        # Generate query embedding
        query_embedding = getEmbeddings(query_text)
        results = localVectorSearch(query_embedding, top_k=top_k)

        # Label results
        labeled_results = []
        for score, doc in results:
            doc_id = str(doc["_id"])
            label = relevance_dict.get(doc_id, 0)  # Default to 0 if not labeled
            labeled_results.append({
                "product_id": doc_id,
                "score": score,
                "label": label,
                "product_name": doc.get("product", {}).get("product_name", "")
            })

        test_set[query_id] = {
            "query_text": query_text,
            "query_embedding": query_embedding,
            "results": labeled_results
        }

    # Evaluate all metrics
    metrics = evaluate_search_performance(test_set, k_values)

    # Print results
    print("\n=== Human-Labeled Evaluation Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save to disk
    with open("human_evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Human evaluation completed.")


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
    # addEmbeddingField()
    # print("Embeddings updated successfully.")
    # CHUNK_SIZE = 2000          # Process 2000 documents at a time
    # EMBEDDING_BATCH_SIZE = 64   # Generate 64 embeddings per batch
    # #SKIP_EXISTING = True        # Skip documents that already have embeddings
    
    # logger.info("Starting optimized embedding generation...")
    
    # # Add embeddings to all documents
    # addEmbeddingFieldOptimized(
    #     chunk_size=CHUNK_SIZE,
    #     embedding_batch_size=EMBEDDING_BATCH_SIZE
    #     #skip_existing=SKIP_EXISTING
    # )
    
    logger.info("Embedding generation completed successfully!")

    # while True:
    #     userQuery = input("Enter a query for Option Search:")
    #     if userQuery.lower() == 'quit':
    #         break

    #     query_embedding = getEmbeddings(userQuery)
    #     topK_results = localVectorSearch(query_embedding)

    if "--evaluate" in sys.argv:
        logger.info("Starting evaluation process...")
        
        # Step 1: Create test set
        test_set = create_test_set(num_queries=100)
        
        # Step 2: Evaluate performance
        metrics = evaluate_search_performance(test_set)
        
        # Step 3: Display results
        print("\n=== Evaluation Results ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Step 4: Save for future comparison
        with open("evaluation_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Evaluation completed!")
        return
    
    elif "--human-eval" in sys.argv:
        logger.info("Starting human-labeled evaluation...")
        evaluate_from_human_json(filepath="human_queries.json")
        return

if __name__ == "__main__":
    main()
else:
    print("This script is intended to be run as a standalone program.")
