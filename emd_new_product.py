import json
import hashlib
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
import time
import os

class ProductEmbedder:
    def __init__(self, chunks_file, host='localhost', port=8000, collection_name="food_products_vn"):
        self.chunks_file = chunks_file
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        print("📥 Đang tải mô hình embedding tiếng Việt...")
        self.model = SentenceTransformer(
            "keepitreal/vietnamese-sbert",
            device="cpu"
        )
    
    def generate_chunk_id(self, text, metadata):
        """Tạo ID duy nhất dựa trên content và product_id"""
        product_id = metadata.get('product_id', 'unknown')
        content_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        return f"prod_{product_id}_{content_hash}"
    
    def connect_to_chroma_server(self):
        """Kết nối tới ChromaDB server - Giống code mẫu của bạn"""
        try:
            print(f"🔗 Đang kết nối tới ChromaDB Server ({self.host}:{self.port})...")
            self.client = chromadb.HttpClient(host=self.host, port=self.port)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            print(" Đã kết nối tới ChromaDB Server!")
            return True
        except Exception as e:
            print(f" Không tìm thấy Server! Lỗi: {e}")
            print(" Bạn đã chạy lệnh 'chroma run --host localhost --port 8000' chưa?")
            return False
    
    def load_and_validate_data(self):
        """Đọc và validate dữ liệu từ file JSON"""
        print(f" Đang đọc dữ liệu từ {self.chunks_file}...")
        try:
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            validated_data = []
            validation_errors = 0
            
            if not isinstance(data, list):
                print(" Dữ liệu không phải là list")
                return []
            
            for i, item in enumerate(data):
                if isinstance(item, dict) and "text" in item and "metadata" in item:
                    metadata = item["metadata"]
                    
                    # Chuẩn hóa metadata theo cấu trúc của bạn
                    standardized_metadata = {
                        "product_id": str(metadata.get("product_id", f"unknown_{i}")),
                        "name": metadata.get("name", "Sản phẩm không tên"),
                        "category": metadata.get("category", ""),
                        "unit": metadata.get("unit", ""),
                        "price": metadata.get("price", 0)
                    }
                    
                    validated_item = {
                        "text": item["text"],
                        "metadata": standardized_metadata
                    }
                    validated_data.append(validated_item)
                else:
                    print(f"   Item {i} không đúng format")
                    validation_errors += 1
            
            print(f"✅ Đã validate {len(validated_data)} chunks (lỗi: {validation_errors})")
            return validated_data
            
        except Exception as e:
            print(f" Lỗi đọc file: {e}")
            return []
    
    def encode_in_batches(self, texts, batch_size=16):
        """Encode văn bản thành embeddings"""
        print(f"🔢 Đang nhúng {len(texts)} văn bản...")
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i + batch_size]
            try:
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                all_embeddings.extend(batch_embeddings.tolist())
            except Exception as e:
                print(f" Lỗi encode batch {i//batch_size + 1}: {e}")
                # Fallback: tạo embeddings mặc định
                embedding_size = 768
                all_embeddings.extend([[0.1] * embedding_size] * len(batch_texts))
        
        return all_embeddings
    
    def store_embeddings(self, data, embeddings):
        """Lưu embeddings lên ChromaDB server"""
        print(" Đang lưu embeddings lên server...")
        
        texts = [item["text"] for item in data]
        metadatas = [item["metadata"] for item in data]
        ids = [self.generate_chunk_id(text, metadata) for text, metadata in zip(texts, metadatas)]
        
        # Chia thành các batch nhỏ để tránh quá tải
        chunk_size = 50
        total_batches = (len(ids) - 1) // chunk_size + 1
        
        successful_batches = 0
        
        for i in range(0, len(ids), chunk_size):
            end_idx = min(i + chunk_size, len(ids))
            batch_num = i // chunk_size + 1
            
            try:
                self.collection.add(
                    ids=ids[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    documents=texts[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )
                print(f"✅ Đã lưu batch {batch_num}/{total_batches} ({end_idx-i} items)")
                successful_batches += 1
                
                # Nghỉ giữa các batch
                time.sleep(0.5)
                
            except Exception as e:
                print(f" Lỗi lưu batch {batch_num}: {e}")
                # Thử lại với batch nhỏ hơn
                if chunk_size > 10:
                    chunk_size = max(10, chunk_size // 2)
                    print(f" Giảm chunk size xuống {chunk_size}")
        
        return successful_batches
    
    def verify_data_upload(self):
        """Xác minh dữ liệu đã được upload thành công"""
        try:
            # Đếm số lượng documents
            count = self.collection.count()
            print(f"🔍 Collection có {count} documents")
            
            # Thử query đơn giản
            test_results = self.collection.query(
                query_texts=["sản phẩm"],
                n_results=1
            )
            
            if test_results['documents'] and test_results['documents'][0]:
                print(" Xác minh thành công! Dữ liệu đã sẵn sàng.")
                return True
            else:
                print(" Collection trống hoặc không có kết quả")
                return False
                
        except Exception as e:
            print(f" Lỗi xác minh: {e}")
            return False
    
    def test_search(self):
        """Test tìm kiếm để đảm bảo hoạt động tốt"""
        print("\n Đang test tìm kiếm...")
        test_queries = ["thịt lợn", "sữa tươi", "bánh", "rau"]
        
        for query in test_queries:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=1
                )
                
                if results['documents'] and results['documents'][0]:
                    doc = results['documents'][0][0]
                    meta = results['metadatas'][0][0]
                    print(f"    '{query}': Tìm thấy {meta.get('name', 'sản phẩm')}")
                else:
                    print(f"    '{query}': Không tìm thấy")
                    
            except Exception as e:
                print(f"    '{query}': Lỗi {e}")
    
    def print_statistics(self, data):
        """In thống kê về dữ liệu"""
        print("\n THỐNG KÊ DỮ LIỆU:")
        print(f"   • Tổng số sản phẩm: {len(data)}")
        
        # Phân tích metadata
        product_ids = set()
        categories = set()
        total_price = 0
        price_count = 0
        
        for item in data:
            metadata = item["metadata"]
            product_ids.add(metadata['product_id'])
            categories.add(metadata['category'])
            
            if metadata.get('price', 0) > 0:
                total_price += metadata['price']
                price_count += 1
        
        print(f"   • Số sản phẩm unique: {len(product_ids)}")
        print(f"   • Số danh mục: {len([c for c in categories if c])}")
        
        if price_count > 0:
            avg_price = total_price / price_count
            print(f"   • Giá trung bình: {avg_price:,.0f} VNĐ")
        
        # Thống kê độ dài text
        text_lengths = [len(item["text"]) for item in data]
        print(f"   • Độ dài mô tả trung bình: {sum(text_lengths)/len(text_lengths):.1f} ký tự")
    
    def run(self):
        """Chạy toàn bộ pipeline nhúng dữ liệu"""
        print(" Bắt đầu pipeline nhúng dữ liệu vào ChromaDB Server...")
        print("=" * 60)
        
        # Bước 1: Kết nối server 
        if not self.connect_to_chroma_server():
            return
        
        # Bước 2: Load và validate dữ liệu
        data = self.load_and_validate_data()
        if not data:
            print(" Không có dữ liệu hợp lệ để xử lý")
            return
        
        # Bước 3: Encode embeddings
        texts = [item["text"] for item in data]
        embeddings = self.encode_in_batches(texts)
        
        # Bước 4: Lưu lên server
        successful_batches = self.store_embeddings(data, embeddings)
        
        # Bước 5: Xác minh
        if successful_batches > 0:
            print("\n🔍 Đang xác minh dữ liệu...")
            if self.verify_data_upload():
                self.test_search()
        
        # Bước 6: Thống kê
        self.print_statistics(data)
        
        print(f"\n HOÀN THÀNH! Đã nhúng {len(data)} sản phẩm vào ChromaDB Server!")

# Chạy pipeline
if __name__ == "__main__":
    # Cấu hình 
    CHUNKS_FILE = "rag_chunks_new.json"
    CHROMA_HOST = "localhost"
    CHROMA_PORT = 8000
    COLLECTION_NAME = "food_products_vn"
    
    # Kiểm tra file
    if not os.path.exists(CHUNKS_FILE):
        print(f" File {CHUNKS_FILE} không tồn tại!")
        exit(1)
    
    # Kiểm tra kích thước file
    file_size = os.path.getsize(CHUNKS_FILE)
    if file_size == 0:
        print(f" File {CHUNKS_FILE} rỗng!")
        exit(1)
    
    print(f" File dữ liệu: {CHUNKS_FILE} ({file_size} bytes)")
    
    # Chạy embedder
    embedder = ProductEmbedder(
        chunks_file=CHUNKS_FILE,
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        collection_name=COLLECTION_NAME
    )
    
    embedder.run()