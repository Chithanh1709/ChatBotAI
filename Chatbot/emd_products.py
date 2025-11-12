# file: optimized_embed_and_store.py

import json
import os
import hashlib
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
import numpy as np

class ProductEmbedder:
    def __init__(self, chunks_file, db_path, collection_name):
        self.chunks_file = chunks_file
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        # Load model với config tối ưu
        print("📥 Đang tải mô hình embedding tiếng Việt...")
        self.model = SentenceTransformer(
            "keepitreal/vietnamese-sbert",
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        )
    
    def generate_chunk_id(self, text, metadata):
        """Tạo ID duy nhất dựa trên content"""
        content = f"{text}_{metadata.get('product_id', '')}"
        return f"chunk_{hashlib.md5(content.encode()).hexdigest()[:12]}"
    
    def load_and_validate_data(self):
        """Đọc và validate dữ liệu"""
        print(f"📖 Đang đọc dữ liệu từ {self.chunks_file}...")
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Validate structure
        validated_data = []
        for i, item in enumerate(data):
            if isinstance(item, dict) and "text" in item:
                validated_data.append(item)
            else:
                print(f"⚠️  Cảnh báo: Item {i} không đúng format")
        
        print(f"✅ Đã validate {len(validated_data)} chunks")
        return validated_data
    
    def encode_in_batches(self, texts, batch_size=32):
        """Encode với batch processing tối ưu"""
        print("🧠 Đang nhúng văn bản...")
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Quan trọng cho cosine similarity
            )
            all_embeddings.extend(batch_embeddings.tolist())
        
        return all_embeddings
    
    def setup_chroma_db(self):
        """Khởi tạo ChromaDB với config tối ưu"""
        print("💾 Đang thiết lập ChromaDB...")
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Xóa collection cũ nếu cần
        try:
            self.client.delete_collection(name=self.collection_name)
            print("♻️  Đã xóa collection cũ")
        except:
            pass
        
        # Tạo collection mới với optimized settings
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "description": "Food products RAG database"
            }
        )
    
    def store_embeddings(self, data, embeddings):
        """Lưu embeddings với chunking thông minh"""
        print("📤 Đang lưu embeddings...")
        
        texts = [item["text"] for item in data]
        metadatas = [item["metadata"] for item in data]
        ids = [self.generate_chunk_id(text, metadata) for text, metadata in zip(texts, metadatas)]
        
        # Chunk lớn để tránh memory issues
        chunk_size = 1000
        for i in range(0, len(ids), chunk_size):
            end_idx = min(i + chunk_size, len(ids))
            
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            print(f"✅ Đã lưu batch {i//chunk_size + 1}/{(len(ids)-1)//chunk_size + 1}")
    
    def run(self):
        """Chạy toàn bộ pipeline"""
        # Load data
        data = self.load_and_validate_data()
        
        if not data:
            print("❌ Không có dữ liệu hợp lệ")
            return
        
        # Encode
        texts = [item["text"] for item in data]
        embeddings = self.encode_in_batches(texts)
        
        # Setup DB
        self.setup_chroma_db()
        
        # Store
        self.store_embeddings(data, embeddings)
        
        # Statistics
        self.print_statistics(data, embeddings)
    
    def print_statistics(self, data, embeddings):
        """In thống kê về dữ liệu"""
        print("\n📊 THỐNG KÊ DỮ LIỆU:")
        print(f"   • Tổng số chunks: {len(data)}")
        print(f"   • Kích thước embedding: {len(embeddings[0])} dimensions")
        
        # Phân tích metadata
        product_ids = set()
        categories = set()
        for item in data:
            metadata = item["metadata"]
            product_ids.add(metadata.get('product_id', ''))
            categories.add(metadata.get('category', ''))
        
        print(f"   • Số sản phẩm unique: {len([pid for pid in product_ids if pid])}")
        print(f"   • Số categories: {len([c for c in categories if c])}")

# Chạy pipeline
if __name__ == "__main__":
    embedder = ProductEmbedder(
        chunks_file="rag_chunks.json",
        db_path="D:/chroma_food_rag",
        collection_name="food_products_vn"
    )
    embedder.run()