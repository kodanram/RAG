import base64
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, Request, status
from google.cloud import storage
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "severity": "%(levelname)s", "message": "%(message)s"}',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Environment Configuration
GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-004")
MAX_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))  # approximate token/character threshold
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Initialize GCP Clients
storage_client = storage.Client(project=GCP_PROJECT)
vertexai.init(project=GCP_PROJECT, location=GCP_REGION)
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)

app = FastAPI(title="GCS Pub/Sub Ingestion Worker", version="1.0.0")


# --- Text Parsing and Chunking Utilities ---

def extract_pages_from_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Extracts raw text by page using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_data = []
    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        text = page.get_text("text").strip()
        if text:
            pages_data.append({"page_number": page_idx + 1, "text": text})
    doc.close()
    return pages_data


def chunk_text(
    pages_data: List[Dict[str, Any]], 
    chunk_size: int = MAX_CHUNK_SIZE, 
    overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Sliding window chunking preserving page-level attribution.
    Can be replaced or augmented by recursive/hierarchical splitters.
    """
    chunks = []
    chunk_counter = 0

    for page_obj in pages_data:
        text = page_obj["text"]
        page_num = page_obj["page_number"]
        
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Snap to whitespace to avoid breaking words if possible
            if end < text_length:
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunk_counter += 1
                chunks.append({
                    "chunk_id": f"p{page_num}_c{chunk_counter}",
                    "page_number": page_num,
                    "content": chunk_content
                })

            if end >= text_length:
                break
            start = end - overlap if (end - overlap) > start else end

    return chunks


# --- Vertex AI Embedding Service ---

def generate_vertex_embeddings(
    chunks: List[Dict[str, Any]], 
    batch_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Batches texts and calls Vertex AI Embeddings API.
    Uses 'RETRIEVAL_DOCUMENT' task type for ingestion.
    """
    processed_records = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        inputs = [
            TextEmbeddingInput(
                text=item["content"], 
                task_type="RETRIEVAL_DOCUMENT"
            ) 
            for item in batch
        ]
        
        try:
            embeddings = embedding_model.get_embeddings(inputs)
            for item, emb in zip(batch, embeddings):
                processed_records.append({
                    **item,
                    "embedding": emb.values  # List[float] representing the dense vector
                })
        except Exception as e:
            logger.error(f"Vertex AI Embedding generation failed for batch starting at index {i}: {str(e)}")
            raise e

    return processed_records


# --- Pub/Sub Webhook Handler ---

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/")
async def process_pubsub_message(request: Request):
    """
    Pub/Sub Push endpoint for Cloud Storage notifications.
    Cloud Storage OBJECT_FINALIZE events send metadata in the message body.
    """
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Malformed JSON in request: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload")

    message = body.get("message")
    if not message or not isinstance(message, dict):
        logger.warning("Invalid Pub/Sub envelope: 'message' field missing")
        return {"status": "ignored", "reason": "Not a valid Pub/Sub message envelope"}

    # Extract base64-encoded GCS notification data
    encoded_data = message.get("data")
    if not encoded_data:
        logger.warning("Empty Pub/Sub message data payload")
        return {"status": "ignored", "reason": "No data in Pub/Sub message"}

    try:
        decoded_bytes = base64.b64decode(encoded_data)
        gcs_event = json.loads(decoded_bytes.decode("utf-8"))
    except Exception as e:
        logger.error(f"Failed to decode Pub/Sub data payload: {str(e)}")
        return {"status": "ignored", "reason": "Failed to decode Base64 payload"}

    # Extract GCS attributes
    bucket_name = gcs_event.get("bucket")
    object_name = gcs_event.get("name")

    if not bucket_name or not object_name:
        logger.info("Non-GCS storage event or missing bucket/name fields.")
        return {"status": "ignored"}

    # Process only PDF objects
    if not object_name.lower().endswith(".pdf"):
        logger.info(f"Skipping non-PDF file: gs://{bucket_name}/{object_name}")
        return {"status": "skipped", "file": object_name}

    logger.info(f"Starting ingestion for: gs://{bucket_name}/{object_name}")

    try:
        # 1. Download PDF to in-memory bytes
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        pdf_bytes = blob.download_as_bytes()
        logger.info(f"Downloaded {len(pdf_bytes)} bytes from gs://{bucket_name}/{object_name}")

        # 2. Extract text per page
        pages_data = extract_pages_from_pdf(pdf_bytes)
        if not pages_data:
            logger.warning(f"No extractable text found in gs://{bucket_name}/{object_name}")
            return {"status": "empty_document"}

        # 3. Chunk text
        chunks = chunk_text(pages_data)
        logger.info(f"Generated {len(chunks)} text chunks for gs://{bucket_name}/{object_name}")

        # 4. Generate Vertex AI embeddings
        embedded_records = generate_vertex_embeddings(chunks)
        logger.info(f"Successfully generated embeddings for {len(embedded_records)} chunks")

        # 5. Vector Database Upsert Target
        # NOTE: Pass `embedded_records` with document metadata to your Vector DB client:
        # e.g., Pinecone, Vertex AI Vector Search, or Cloud SQL pgvector.
        #
        # upsert_to_vector_db(
        #     vectors=[
        #         {
        #             "id": f"{object_name}#{rec['chunk_id']}",
        #             "values": rec["embedding"],
        #             "metadata": {
        #                 "source_gcs_uri": f"gs://{bucket_name}/{object_name}",
        #                 "page": rec["page_number"],
        #                 "text": rec["content"]
        #             }
        #         } for rec in embedded_records
        #     ]
        # )

        return {
            "status": "success",
            "bucket": bucket_name,
            "object": object_name,
            "chunks_processed": len(embedded_records)
        }

    except Exception as e:
        logger.error(f"Error during ingestion of gs://{bucket_name}/{object_name}: {str(e)}", exc_info=True)
        # Raising 500 signals Pub/Sub to retry according to its retry/dead-letter policy
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Worker processing failed: {str(e)}"
        )