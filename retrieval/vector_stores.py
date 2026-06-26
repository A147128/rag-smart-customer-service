from langchain_chroma import Chroma

from config import config_data as config


class VectorStoreService:
    def __init__(self, embedding) -> None:
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self, k: int | None = None):
        """Return a Chroma retriever."""
        return self.vector_store.as_retriever(search_kwargs={"k": k or config.retrieval_top_k})

    def get_retriever_with_scores(self, k: int | None = None):
        """Return a retriever that preserves Chroma similarity scores."""
        return self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k or config.retrieval_top_k, "score_threshold": 0.0},
        )


if __name__ == "__main__":
    from langchain_community.embeddings import DashScopeEmbeddings

    retriever = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()
    res = retriever.invoke("我的体重180斤，尺码推荐")
    print(res)
