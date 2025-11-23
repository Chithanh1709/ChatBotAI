from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DB_PATH = "D:/chroma_food_rag"
COLLECTION_NAME = "food_products_vn"
EMBED_MODEL = "keepitreal/vietnamese-sbert"


def add_product_to_vector_db(product_id: str, title: str, description: str):
    """
    Thêm sản phẩm vào Chroma DB theo đúng chuẩn LangChain.
    """

    # 1. Load embedding giống với hệ thống query
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    # 2. Load Chroma database
    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # 3. Tạo document
    content = f"{title}\n{description}"
    metadata = {"product_id": product_id, "title": title}

    # 4. Thêm document vào DB
    vector_store.add_texts(
        texts=[content],
        metadatas=[metadata],
        ids=[product_id]  # dùng product_id làm identifier
    )

    print(f"✔ Added product '{product_id}' to Chroma DB")


# -----------------------
# Example usage:
# -----------------------
if __name__ == "__main__":
    add_product_to_vector_db(
        product_id="SP002",
        title="Gạo ST2",
        description="Gạo thơm, hạt dài, ngon nhất Việt."
    )