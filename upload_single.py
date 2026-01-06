from pinecone import ServerlessSpec

# === Ensure Pinecone index exists ===
existing_indexes = [idx["name"] for idx in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")

    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,  # Gemini embedding-001 dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Safe to connect now
index = pc.Index(PINECONE_INDEX_NAME)
