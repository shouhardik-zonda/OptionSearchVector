from sentence_transformers import SentenceTransformer
from pymongo import MongoClient, UpdateOne
import numpy as np
import gc
import torch
from tqdm import tqdm
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = MongoClient("mongodb://localhost:27017/")
db = client["local"]
collection = db["optionProducts"]

def getEmbeddingFields(document):
    """Extract text fields for embedding generation"""
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

def initializeModel():
    """Initialize the sentence transformer model with optimizations"""
    model = SentenceTransformer("intfloat/e5-base-v2", device="mps")
    
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

def addEmbeddingFieldOptimized(chunk_size=2000, embedding_batch_size=64, skip_existing=True):
    """
    Optimized function to add embedding fields to documents
    
    Args:
        chunk_size: Number of documents to process in each chunk
        embedding_batch_size: Batch size for embedding generation
        skip_existing: Skip documents that already have embeddings
    """
    
    model = initializeModel()
    
    # Get total count for progress tracking
    filter_query = {"embedding": {"$exists": False}} if skip_existing else {}
    docs_to_process = collection.count_documents(filter_query)
    logger.info(f"Documents needing embedding: {docs_to_process}")
    
    if docs_to_process == 0:
        logger.info("All documents already have embeddings!")
        return
    
    processed_count = 0
    
    # Process in chunks
    with tqdm(total=docs_to_process, desc="Processing embeddings") as pbar:
        skip = 0
        
        while skip < docs_to_process:
            try:
                # Fetch chunk of documents
                cursor = collection.find(filter_query).skip(skip).limit(chunk_size)
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
                
                if not texts:
                    skip += len(documents)
                    continue
                
                logger.info(f"Processing chunk: {len(texts)} documents")
                
                # Generate embeddings in batches
                embeddings = processBatchEmbeddings(texts, model, embedding_batch_size)
                
                # Prepare bulk update operations
                bulk_operations = []
                for doc_id, embedding in zip(doc_ids, embeddings):
                    bulk_operations.append(
                        UpdateOne(
                            {"_id": doc_id},
                            {"$set": {"embedding": embedding}},
                            upsert=False
                        )
                    )
                
                # Execute bulk write
                if bulk_operations:
                    result = collection.bulk_write(bulk_operations, ordered=False)
                    logger.info(f"Updated {result.modified_count} documents")
                    processed_count += result.modified_count
                
                # Clear memory
                del embeddings, bulk_operations, documents, texts
                gc.collect()
                
                # Update progress
                pbar.update(len(doc_ids))
                skip += chunk_size
                
                # Add small delay to prevent overwhelming the system
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing chunk at skip {skip}: {e}")
                skip += chunk_size
                continue
    
    logger.info(f"Completed! Processed {processed_count} documents total")

def main():
    """Main function with optimized parameters"""
    # Optimized parameters for 320k documents
    CHUNK_SIZE = 2000          # Process 2000 documents at a time
    EMBEDDING_BATCH_SIZE = 64   # Generate 64 embeddings per batch
    SKIP_EXISTING = True        # Skip documents that already have embeddings
    
    logger.info("Starting optimized embedding generation...")
    
    # Add embeddings to all documents
    addEmbeddingFieldOptimized(
        chunk_size=CHUNK_SIZE,
        embedding_batch_size=EMBEDDING_BATCH_SIZE,
        skip_existing=SKIP_EXISTING
    )
    
    logger.info("Embedding generation completed successfully!")

if __name__ == "__main__":
    main()
