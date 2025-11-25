from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Config
DB_PATH = "D:/chroma_food_rag"
COL = "food_products_vn"
emb = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert", model_kwargs={"device":"cpu"})

def debug_database():
    """Debug toàn bộ database"""
    vecdb = Chroma(persist_directory=DB_PATH, embedding_function=emb, collection_name=COL)
    
    # Lấy tất cả data
    all_data = vecdb.get()
    total_docs = len(all_data['documents'])
    
    print(f"📊 Tổng số documents: {total_docs}")
    print("=" * 60)
    
    # Hiển thị 5 documents đầu tiên
    print("📝 5 DOCUMENTS ĐẦU TIÊN:")
    for i in range(min(5, total_docs)):
        print(f"\n--- Document {i+1} ---")
        print(f"ID: {all_data['ids'][i]}")
        print(f"Content: {all_data['documents'][i][:200]}...")
        print(f"Metadata: {all_data['metadatas'][i]}")
    
    # Test search
    print("\n" + "=" * 60)
    print("🔍 TEST SEARCH:")
    
    test_queries = ["cải thảo", "rau", "sữa", "bánh"]
    for query in test_queries:
        print(f"\nTìm kiếm: '{query}'")
        results = vecdb.similarity_search(query, k=2)
        print(f"Found: {len(results)} results")
        for j, doc in enumerate(results):
            print(f"  {j+1}. {doc.metadata.get('name', 'No name')} - {doc.page_content[:100]}...")

if __name__ == "__main__":
    debug_database()