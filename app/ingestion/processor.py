import hashlib
from pathlib import Path

import logfire
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.ingestion.chunking.splitter import chunk_text
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.text import parse_text
from app.services.retrieval.embeddings import embed_text, get_embedding_dim


# ---------------------------------------------------------
# Logfire
# ---------------------------------------------------------

logfire.configure(
    service_name="rag-ingestion-service"
)


# ---------------------------------------------------------
# Qdrant
# ---------------------------------------------------------

qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)

collection_name = settings.QDRANT_COLLECTION


# ---------------------------------------------------------
# Supported file types
# ---------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".html",
    ".htm",
    ".docx",
    ".pptx",
}


# ---------------------------------------------------------
# Create Qdrant collection if it doesn't exist
# ---------------------------------------------------------

if not qdrant_client.collection_exists(collection_name):

    with logfire.span("Create Qdrant Collection"):

        embedding_dim = get_embedding_dim()

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        )

        logfire.info(
            "Qdrant collection created",
            collection=collection_name,
            embedding_dimension=embedding_dim,
            distance="cosine",
        )


# ---------------------------------------------------------
# Generate deterministic document ID
# ---------------------------------------------------------

def generate_document_id(file_path: str) -> str:
    """
    Generate a stable document ID from the file path.

    The same file path will always generate the same ID.
    """

    normalized_path = str(Path(file_path).resolve())

    return hashlib.sha256(
        normalized_path.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------
# Generate deterministic chunk ID
# ---------------------------------------------------------

import uuid

def generate_chunk_id(
    document_id: str,
    chunk_index: int,
) -> str:
    """Generate a stable Qdrant point ID for a chunk."""

    chunk_key = f"{document_id}:{chunk_index}"

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            chunk_key,
        )
    )


# ---------------------------------------------------------
# Process a single file
# ---------------------------------------------------------

def process_file(file_path: str,filename: str,source_type: str,) -> int:
    """ Parse → chunk → embed → index in Qdrant."""

    with logfire.span(
        "Processing File",
        file=filename,
        source=source_type,
    ):

        try:

            # -------------------------------------------------
            # 1. Parse document
            # -------------------------------------------------

            with logfire.span(
                "Parse Document",
                file=filename,
            ):

                ext = Path(filename).suffix.lower()

                if ext == ".pdf":

                    full_text = parse_pdf(file_path)

                elif ext in (".html", ".htm"):

                    full_text = parse_html(file_path)

                elif ext == ".txt":

                    full_text = parse_text(file_path)

                elif ext in (".docx", ".pptx"):

                    from app.ingestion.loaders.office import parse_office

                    full_text = parse_office(file_path)

                else:

                    logfire.warning(
                        "Unsupported file type",
                        file=filename,
                        extension=ext,
                    )

                    return 0

            # -------------------------------------------------
            # 2. Validate extracted text
            # -------------------------------------------------

            if not full_text or not full_text.strip():

                logfire.warning(
                    "No text extracted",
                    file=filename,
                )

                return 0

            # -------------------------------------------------
            # 3. Chunk document
            # -------------------------------------------------

            with logfire.span(
                "Chunk Document",
                file=filename,
            ):

                chunks = chunk_text(full_text)

            if not chunks:

                raise ValueError(
                    f"No chunks created from: {filename}"
                )

            logfire.info(
                "Document chunked",
                file=filename,
                chunk_count=len(chunks),
            )

            # -------------------------------------------------
            # 4. Generate embeddings
            # -------------------------------------------------

            with logfire.span(
                "Generate Embeddings",
                file=filename,
                chunk_count=len(chunks),
            ):

                embeddings = embed_text(chunks)

            # Validate one-to-one mapping

            if len(embeddings) != len(chunks):

                raise ValueError(
                    f"Embedding count ({len(embeddings)}) "
                    f"does not match chunk count ({len(chunks)})"
                )

            # -------------------------------------------------
            # 5. Generate document ID
            # -------------------------------------------------

            document_id = generate_document_id(file_path)

            # -------------------------------------------------
            # 6. Create Qdrant points
            # -------------------------------------------------

            with logfire.span(
                "Create Qdrant Points",
                file=filename,
                chunk_count=len(chunks),
            ):

                points = []

                for chunk_index, (chunk, vector) in enumerate(
                    zip(chunks, embeddings)
                ):

                    point_id = generate_chunk_id(
                        document_id=document_id,
                        chunk_index=chunk_index,
                    )

                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={
                                "text": chunk,
                                "source": filename,
                                "source_type": source_type,
                                "document_id": document_id,
                                "chunk_index": chunk_index,
                            },
                        )
                    )

            # -------------------------------------------------
            # 7. Upsert into Qdrant
            # -------------------------------------------------

            with logfire.span(
                "Qdrant Upsert",
                file=filename,
                point_count=len(points),
            ):

                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points,
                )

            logfire.info(
                "File indexed successfully",
                file=filename,
                document_id=document_id,
                chunks_indexed=len(points),
            )

            return len(points)

        except Exception as e:

            logfire.error(
                "File processing failed",
                file=filename,
                error=str(e),
            )

            # Important:
            # Re-raise so process_directory()
            # knows that this file actually failed.
            raise


# ---------------------------------------------------------
# Process entire directory
# ---------------------------------------------------------

def process_directory(directory: str):
    """
    Recursively process all supported files
    inside a directory.
    """

    directory_path = Path(directory)

    # -----------------------------------------------------
    # Validate directory
    # -----------------------------------------------------

    if not directory_path.exists():

        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    if not directory_path.is_dir():

        raise ValueError(
            f"Path is not a directory: {directory}"
        )

    # -----------------------------------------------------
    # Find files
    # -----------------------------------------------------

    files = [
        file_path
        for file_path in directory_path.rglob("*")
        if file_path.is_file()
    ]

    logfire.info(
        "Files discovered",
        directory=str(directory_path),
        total_files=len(files),
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    processed = 0
    skipped = 0
    failed = 0
    total_chunks = 0

    # -----------------------------------------------------
    # Process files
    # -----------------------------------------------------

    with logfire.span(
        "Directory Ingestion",
        directory=str(directory_path),
        total_files=len(files),
    ):

        for file_path in files:

            extension = file_path.suffix.lower()

            # ---------------------------------------------
            # Skip unsupported files
            # ---------------------------------------------

            if extension not in SUPPORTED_EXTENSIONS:

                logfire.warning(
                    "Skipping unsupported file",
                    file=str(file_path),
                    extension=extension,
                )

                skipped += 1

                continue

            # ---------------------------------------------
            # Process file
            # ---------------------------------------------

            try:

                logfire.info(
                    "Starting file processing",
                    file=str(file_path),
                )

                chunks_indexed = process_file(
                    file_path=str(file_path),
                    filename=file_path.name,
                    source_type=extension.lstrip("."),
                )

                if chunks_indexed == 0:

                    skipped += 1

                    continue

                processed += 1
                total_chunks += chunks_indexed

                logfire.info(
                    "File processing completed",
                    file=file_path.name,
                    chunks_indexed=chunks_indexed,
                )

            except Exception as e:

                failed += 1

                logfire.error(
                    "File processing failed",
                    file=file_path.name,
                    error=str(e),
                )

                # Continue processing remaining files
                continue

    # -----------------------------------------------------
    # Final ingestion summary
    # -----------------------------------------------------

    logfire.info(
        "Directory ingestion completed",
        directory=str(directory_path),
        total_files=len(files),
        processed=processed,
        skipped=skipped,
        failed=failed,
        total_chunks=total_chunks,
    )

    return {
        "total_files": len(files),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "total_chunks": total_chunks,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    result = process_directory(
        "data/knowledge_base"
    )

    print("\nIngestion completed")
    print("-------------------")
    print(f"Total files : {result['total_files']}")
    print(f"Processed   : {result['processed']}")
    print(f"Skipped     : {result['skipped']}")
    print(f"Failed      : {result['failed']}")
    print(f"Total chunks: {result['total_chunks']}")