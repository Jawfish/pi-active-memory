export class TransactionalVectorStoreError extends Error {
  readonly code = "ACTIVE_MEMORY_VECTOR_STORE_V2_REQUIRED";

  constructor() {
    super("Transactional VectorStore v2 required: migrate the RAG adapter to contractVersion: 2 with transactional mutate/compact/rebuildVectors methods");
    this.name = "TransactionalVectorStoreError";
  }
}
