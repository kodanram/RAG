# GCS PDF Embedding Ingestion Worker

This project provides a FastAPI service that processes PDF files uploaded to
Google Cloud Storage (GCS). A GCS notification is delivered through a
Pub/Sub push subscription, and the service:

1. Downloads the PDF from GCS.
2. Extracts text page by page using PyMuPDF.
3. Splits the text into overlapping chunks.
4. Generates Vertex AI embeddings for each chunk.
5. Returns processing metadata for the document.

The generated embeddings are currently kept in memory for the request. The
vector database upsert is provided as a commented integration point and must
be implemented for a complete retrieval-augmented generation (RAG) pipeline.

## Architecture

```text
PDF uploaded to GCS
        |
        v
Cloud Storage notification
        |
        v
Pub/Sub push subscription
        |
        v
FastAPI POST /
        |
        +--> Download PDF from GCS
        +--> Extract page text
        +--> Create overlapping chunks
        +--> Generate Vertex AI embeddings
        +--> Upsert to a vector database (to be implemented)
```

## Project structure

```text
genai-project/
└── main.py       # FastAPI application and ingestion pipeline
```

## Requirements

- Python 3.10 or later
- A Google Cloud project
- A GCS bucket containing the PDF documents
- Vertex AI enabled in the selected region
- A Pub/Sub topic and push subscription
- Google Cloud Application Default Credentials

The application uses the following Python packages:

- FastAPI
- Uvicorn
- PyMuPDF
- `google-cloud-storage`
- `google-cloud-aiplatform`

Install them with:

```powershell
pip install fastapi uvicorn pymupdf google-cloud-storage google-cloud-aiplatform
```

## Google Cloud authentication

The service uses Application Default Credentials. For local development,
authenticate with:

```powershell
gcloud auth application-default login
```

When deployed to Google Cloud, assign the runtime service account permissions
to:

- Read objects from the source GCS bucket.
- Call Vertex AI prediction APIs.

## Configuration

Configure the service with environment variables:

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `GCP_PROJECT` | Yes | None | Google Cloud project ID |
| `GCP_REGION` | No | `us-central1` | Vertex AI region |
| `EMBEDDING_MODEL_NAME` | No | `text-embedding-004` | Vertex AI embedding model |
| `CHUNK_SIZE` | No | `600` | Approximate chunk size in characters |
| `CHUNK_OVERLAP` | No | `100` | Number of overlapping characters between chunks |

Example PowerShell configuration:

```powershell
$env:GCP_PROJECT = "your-gcp-project-id"
$env:GCP_REGION = "us-central1"
$env:EMBEDDING_MODEL_NAME = "text-embedding-004"
$env:CHUNK_SIZE = "600"
$env:CHUNK_OVERLAP = "100"
```

The Google Cloud clients and Vertex AI embedding model are initialized when
the application starts, so `GCP_PROJECT` and valid credentials should be
available before launching the service.

## Running locally

Start the FastAPI application with Uvicorn:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8080
```

The interactive API documentation is available at:

```text
http://localhost:8080/docs
```

## Endpoints

### `GET /healthz`

Health-check endpoint used by deployment platforms and monitoring systems.

Response:

```json
{
  "status": "ok"
}
```

### `POST /`

Pub/Sub push endpoint for GCS object notifications.

The endpoint expects a Pub/Sub envelope whose `message.data` field contains
Base64-encoded JSON describing the GCS object. A minimal example is:

```json
{
  "message": {
    "data": "eyJidWNrZXQiOiAibXktcGRmLWJ1Y2tldCIsICJuYW1lIjogImRvY3MvaW52b2ljZS5wZGYifQ=="
  }
}
```

The decoded event must include:

```json
{
  "bucket": "my-pdf-bucket",
  "name": "docs/invoice.pdf"
}
```

Only objects whose names end in `.pdf` are processed. Other objects are
skipped.

Successful response:

```json
{
  "status": "success",
  "bucket": "my-pdf-bucket",
  "object": "docs/invoice.pdf",
  "chunks_processed": 12
}
```

## Processing details

### PDF extraction

PDF bytes are downloaded directly into memory. PyMuPDF extracts text from
each page, and pages without extractable text are ignored. The page number is
preserved as metadata for every generated chunk.

### Chunking

Text is processed with a sliding window. The chunk boundary is moved to the
last whitespace character when possible to avoid splitting words. Overlap
helps preserve context between adjacent chunks.

Each chunk has this shape:

```json
{
  "chunk_id": "p2_c4",
  "page_number": 2,
  "content": "Extracted text from the PDF..."
}
```

`CHUNK_SIZE` and `CHUNK_OVERLAP` are character-based values in the current
implementation, despite the chunk-size comment referring to tokens.

### Embeddings

Chunks are sent to Vertex AI in batches of up to 100 inputs using the
`RETRIEVAL_DOCUMENT` task type. Each chunk receives a dense vector:

```json
{
  "chunk_id": "p2_c4",
  "page_number": 2,
  "content": "Extracted text from the PDF...",
  "embedding": [0.012, -0.034]
}
```

## Response and error behavior

| Condition | Response |
| --- | --- |
| Valid PDF processed | `status: success` |
| Non-PDF object | `status: skipped` |
| PDF contains no extractable text | `status: empty_document` |
| Missing Pub/Sub message envelope | `status: ignored` |
| Missing Pub/Sub data | `status: ignored` |
| Invalid Base64 or event JSON | `status: ignored` |
| Invalid HTTP JSON body | HTTP `400` |
| Download, parsing, or embedding failure | HTTP `500` |

An HTTP `500` is raised for processing failures so that Pub/Sub can retry the
message according to its retry and dead-letter configuration.

## Vector database integration

The application currently stops after embedding generation. To persist the
vectors, implement the upsert section in `main.py` for a vector database such
as:

- Vertex AI Vector Search
- Pinecone
- Cloud SQL with `pgvector`

The intended vector record includes:

```python
{
    "id": f"{object_name}#{chunk_id}",
    "values": embedding,
    "metadata": {
        "source_gcs_uri": f"gs://{bucket_name}/{object_name}",
        "page": page_number,
        "text": content,
    },
}
```

## Limitations and production considerations

- Scanned or image-only PDFs require OCR, which is not included.
- Vector database persistence is not implemented yet.
- Pub/Sub authentication or push-token validation is not implemented in the
  application and should be configured at the deployment or gateway layer.
- Pub/Sub may deliver duplicate messages, so production persistence should be
  idempotent.
- The service processes documents in memory; large PDFs may require memory
  limits and additional safeguards.
- The current chunking threshold is based on characters rather than model
  tokens.

