import logfire
from sentence_transformers import SentenceTransformer

BATCH_SIZE = 50


_model : SentenceTransformer | None = None

# ---------------------- Model Initialization -----------------------

def _init():
    """Initialize the embedding model once per process."""
    global _model

    if _model is not None:
        return 

    logfire.info(
        "Loading Qwen3-Embedding-0.6B (1024-dim)."
    ) 

    _model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B"
    )

    logfire.info(
        "Qwen3-Embedding-0.6B loaded successfully."
    )


# ---------------------- Public Helpers -----------------

def get_embedding_dim()->int:
    _init()

    dim = _model.get_embedding_dimension()

    if dim is None:
        raise RuntimeError("Could not determine embedding dimension")

    return dim

# ---------------------- Batch Embedding --------------------
def _embed_batch(batch : list[str])->list[list[float]]:
    """Embed a batch of texts using Qwen3-Embedding-0.6B"""
    return _model.encode( 
        batch, show_progress_bar=False, 
        normalize_embeddings=True
        ).tolist()


# ---------------------- Embed Query ------------------------
def embed_query(query:str) -> list[float]:
    """Generate an embedding for a single query"""

    _init()

    return _model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

# ----------------------- Embed Batch ------------------------
def embed_text(texts : list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts in batches."""
    _init()

    all_embeddings : list[list[float]] = []

    for i in range(0,len(texts),BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]

        with logfire.span(
            "Embed batch",
            model="Qwen/Qwen3-Embedding-0.6B",
            start=i,
            size=len(batch),
        ):
            all_embeddings.extend(
                _embed_batch(batch)
            )

    return all_embeddings