import chromadb
import requests
import json
import logging
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MEGALLM_BASE_URL = "https://ai.megallm.io/v1"
MEGALLM_API_KEY = "sk-mega-9e02941cc7286047dfe1dc53d2d94a1afddddd677e4769b5189ed82a992f2f43"  
MEGALLM_MODEL_NAME = "llama3.3-70b-instruct"  

CHROMA_SERVER_HOST = "localhost"
CHROMA_SERVER_PORT = 8000
COLLECTION_NAME = "food_products_vn"
PRODUCT_BASE_URL = "http://localhost:4200/product"

class ChromaRAGSystem:
    def __init__(self):
        self.client = None
        self.collection = None
        self.megallm_ready = False
        self.embedding_model = None
        self.initialize_system()

    def initialize_system(self):
        print(" Đang khởi tạo hệ thống RAG...")
        
        print(" Đang tải mô hình embedding...")
        self.embedding_model = SentenceTransformer(
            "keepitreal/vietnamese-sbert",
            device="cpu"
        )
        
        try:
            self.client = chromadb.HttpClient(
                host=CHROMA_SERVER_HOST, 
                port=CHROMA_SERVER_PORT
            )
            
            self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
            print(" Đã kết nối tới ChromaDB Server!")
            
            count = self.collection.count()
            print(f" Database có {count} sản phẩm")
            
        except Exception as e:
            print(f" Lỗi kết nối ChromaDB: {e}")
            return

        self.megallm_ready = self.initialize_megallm_client()

    def initialize_megallm_client(self):
        try:
            headers = {
                "Authorization": f"Bearer {MEGALLM_API_KEY}",
                "Content-Type": "application/json"
            }
            
            test_payload = {
                "model": MEGALLM_MODEL_NAME,
                "messages": [{"role": "user", "content": "Xin chào"}],
                "max_tokens": 50
            }
            
            response = requests.post(
                f"{MEGALLM_BASE_URL}/chat/completions", 
                json=test_payload, 
                headers=headers, 
                timeout=30
            )
            
            if response.status_code == 200:
                print("✅ Đã kết nối thành công với MegaLLM")
                return True
            else:
                print(f"⚠️ Lỗi kết nối MegaLLM: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"⚠️ Lỗi cấu hình MegaLLM: {e}")
            return False

    def classify_query_with_llm(self, query: str) -> Tuple[bool, str]:
        """Sử dụng LLM để phân loại câu hỏi có liên quan đến thực phẩm không"""
        system_message = """Bạn là hệ thống phân loại câu hỏi. Hãy phân tích câu hỏi và xác định xem nó có liên quan đến thực phẩm, đồ uống, sản phẩm ăn uống không.

PHẠM VI THỰC PHẨM BAO GỒM:
- Thực phẩm, đồ ăn, thức uống
- Nguyên liệu nấu ăn
- Thành phần dinh dưỡng
- Cách bảo quản thực phẩm
- Thông tin sản phẩm ăn uống
- Giá cả, đặc tính sản phẩm thực phẩm

PHẠM VI KHÔNG BAO GỒM:
- Câu hỏi về y tế, thuốc men
- Câu hỏi về công nghệ, xe cộ
- Câu hỏi về thời tiết, tin tức
- Câu hỏi chung chung không liên quan

TRẢ LỜI THEO ĐỊNH DẠNG JSON:
{
    "is_food_related": true/false,
    "reason": "lý do ngắn gọn"
}"""

        user_content = f"CÂU HỎI CẦN PHÂN LOẠI: {query}"

        try:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ]
            
            payload = {
                "model": MEGALLM_MODEL_NAME,
                "messages": messages,
                "max_tokens": 200,
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            
            headers = {
                "Authorization": f"Bearer {MEGALLM_API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{MEGALLM_BASE_URL}/chat/completions", 
                json=payload, 
                headers=headers, 
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                classification_text = result["choices"][0]["message"]["content"]
                
                try:
                    classification = json.loads(classification_text)
                    is_food_related = classification.get("is_food_related", False)
                    reason = classification.get("reason", "Không xác định")
                    return is_food_related, reason
                except json.JSONDecodeError:
                    # Fallback: phân tích đơn giản nếu LLM không trả về JSON
                    return self.fallback_classification(query), "Phân tích fallback"
                    
            else:
                logger.error(f"Lỗi phân loại LLM: {response.status_code}")
                return self.fallback_classification(query), "Lỗi kết nối LLM"
                
        except Exception as e:
            logger.error(f"Lỗi phân loại: {e}")
            return self.fallback_classification(query), "Lỗi hệ thống"

    def fallback_classification(self, query: str) -> bool:
        """Phân loại fallback đơn giản khi LLM không hoạt động"""
        query_lower = query.lower().strip()
        
        # Các từ khóa cơ bản để tránh các câu hỏi hoàn toàn không liên quan
        unrelated_keywords = [
            "thời tiết", "xe", "máy tính", "điện thoại", "y tế", "bác sĩ", "bệnh",
            "thuốc", "chính trị", "thể thao", "bóng đá", "âm nhạc", "phim ảnh",
            "du lịch", "khách sạn", "ngân hàng", "tiền", "chứng khoán", "công nghệ"
        ]
        
        # Nếu có từ khóa hoàn toàn không liên quan -> không phải thực phẩm
        if any(keyword in query_lower for keyword in unrelated_keywords):
            return False
            
        # Mặc định cho phép tìm kiếm để tránh bỏ sót câu hỏi hợp lệ
        return True

    def encode_query(self, query: str):
        try:
            embedding = self.embedding_model.encode([query])
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Lỗi encode query: {e}")
            return None

    def search_products(self, query: str, n_results: int = 3):
        try:
            query_embedding = self.encode_query(query)
            if query_embedding is None:
                return None
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            return results
        except Exception as e:
            logger.error(f"Lỗi tìm kiếm: {e}")
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                return results
            except Exception as e2:
                logger.error(f"Lỗi tìm kiếm fallback: {e2}")
                return None

    def get_product_id(self, metadata: dict) -> str:
        possible_id_fields = ["id", "product_id", "productId", "ID", "productID"]
        
        for field in possible_id_fields:
            product_id = metadata.get(field)
            if product_id and product_id != "unknown":
                return str(product_id)
        
        return "unknown"

    def format_context_for_megallm(self, results: Dict) -> str:
        if not results or not results['documents']:
            return "Không có thông tin sản phẩm."

        context = "THÔNG TIN SẢN PHẨM TÌM THẤY:\n\n"
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            product_id = self.get_product_id(metadata)
            context += f"=== SẢN PHẨM {i} ===\n"
            context += f"ID: {product_id}\n"
            context += f"Tên: {metadata.get('name', 'Chưa có tên')}\n"
            
            if metadata.get("price") and metadata["price"] > 0:
                context += f"Giá: {metadata['price']:,} VNĐ\n"
            if metadata.get("category"):
                context += f"Danh mục: {metadata['category']}\n"
            if metadata.get("unit"):
                context += f"Đơn vị: {metadata['unit']}\n"
                
            context += f"Mô tả: {doc}\n\n"
            
        return context

    def create_megallm_prompt(self, context: str, question: str) -> List[dict]:
        system_message = """Bạn là chuyên gia tư vấn thực phẩm. Hãy sử dụng thông tin sản phẩm được cung cấp để trả lời câu hỏi.

QUY TẮC TRẢ LỜI:
- CHỈ sử dụng thông tin được cung cấp trong THÔNG TIN SẢN PHẨM
- KHÔNG được bịa thêm thông tin không có trong dữ liệu
- Nếu không có thông tin phù hợp, hãy nói rõ "Không tìm thấy thông tin phù hợp trong cơ sở dữ liệu"
- Trả lời bằng tiếng Việt tự nhiên, thân thiện
- Tập trung vào thông tin thực tế về sản phẩm
- Khi đề cập đến sản phẩm, có thể tham khảo ID và tên sản phẩm"""

        user_content = f"""DỮ LIỆU SẢN PHẨM:
{context}

CÂU HỎI CỦA NGƯỜI DÙNG: {question}

Dựa trên thông tin sản phẩm trên, hãy trả lời câu hỏi:"""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

    def ask_megallm(self, context: str, question: str) -> str:
        try:
            messages = self.create_megallm_prompt(context, question)
            
            payload = {
                "model": MEGALLM_MODEL_NAME,
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.3,
                "top_p": 0.9
            }
            
            headers = {
                "Authorization": f"Bearer {MEGALLM_API_KEY}",
                "Content-Type": "application/json"
            }
            
            print(" Đang tạo câu trả lời với AI...")
            response = requests.post(
                f"{MEGALLM_BASE_URL}/chat/completions", 
                json=payload, 
                headers=headers, 
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"MegaLLM API error: {response.status_code} - {response.text}")
                return f"Xin lỗi, tôi gặp sự cố khi kết nối với AI. Lỗi: {response.status_code}"
                
        except Exception as e:
            logger.error(f"MegaLLM connection error: {e}")
            return f"Xin lỗi, tôi gặp sự cố kết nối. Vui lòng thử lại sau."

    def generate_product_link(self, product_id: str) -> str:
        if product_id and product_id != "unknown":
            return f"{PRODUCT_BASE_URL}/{product_id}"
        return "Không có link"

    def smart_fallback_response(self, results: Dict, question: str) -> str:
        if not results or not results['documents']:
            return " Không tìm thấy sản phẩm nào phù hợp với yêu cầu của bạn."

        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        response = f"🔍 Tìm thấy {len(docs)} sản phẩm liên quan đến '{question}':\n\n"
        
        for i, (doc, metadata) in enumerate(zip(docs, metadatas), 1):
            product_id = self.get_product_id(metadata)
            product_link = self.generate_product_link(product_id)
            
            response += f" {metadata.get('name', 'Sản phẩm')}\n"
            response += f"    Mô tả: {doc[:100]}...\n"
            
            if metadata.get("price") and metadata["price"] > 0:
                response += f"    Giá: {metadata['price']:,} VNĐ\n"
            if metadata.get("category"):
                response += f"    Danh mục: {metadata['category']}\n"
            if product_link != "Không có link":
                response += f"   🔗 Link sản phẩm: {product_link}\n"
                
            response += "\n"

        return response

    def get_answer(self, question: str) -> str:
        try:
            if question is None:
                return " Câu hỏi không hợp lệ."
                
            question_str = str(question).strip() if question else ""
            
            if not question_str:
                return " Bạn muốn tìm hiểu về sản phẩm nào? Hãy nhập câu hỏi cụ thể."
                
            if len(question_str) < 2:
                return " Vui lòng nhập câu hỏi rõ ràng hơn (ít nhất 2 ký tự)."
                
            logger.info(f"🔍 Người dùng hỏi: '{question_str}'")
            
            if not self.collection:
                return " Hệ thống đang được bảo trì. Vui lòng thử lại sau."

            # Sử dụng LLM để phân loại câu hỏi
            print(" Đang phân tích câu hỏi...")
            is_food_related, reason = self.classify_query_with_llm(question_str)
            logger.info(f"Phân loại: {is_food_related} - Lý do: {reason}")
            
            if not is_food_related:
                return (
                    f" Câu hỏi của bạn không thuộc phạm vi tư vấn thực phẩm.\n"
                    f" Lý do: {reason}\n\n"
                    f" Tôi chuyên tư vấn về:\n"
                    f"   • Thực phẩm, đồ ăn, thức uống\n"
                    f"   • Nguyên liệu nấu ăn\n"
                    f"   • Thành phần dinh dưỡng\n"
                    f"   • Thông tin sản phẩm ăn uống\n\n"
                    f" Ví dụ câu hỏi phù hợp:\n"
                    f"   - 'Sữa tươi nào tốt cho trẻ em?'\n"
                    f"   - 'Thành phần của bánh gạo là gì?'\n"
                    f"   - 'Có loại thịt lợn hữu cơ không?'\n"
                    f"   - 'Giá phô mai Mozzarella bao nhiêu?'"
                )

            logger.info(" Đang tìm kiếm trong cơ sở dữ liệu...")
            results = self.search_products(question_str, n_results=3)
            
            if not results or not results['documents'] or not results['documents'][0]:
                return (
                    " Không tìm thấy sản phẩm phù hợp.\n\n"
                    " Gợi ý:\n"
                    "   • Kiểm tra lại chính tả\n"
                    "   • Thử từ khóa khác\n"
                    "   • Mô tả cụ thể hơn\n\n"
                    " Ví dụ:\n"
                    "   - 'Thịt lợn ba chỉ'\n"
                    "   - 'Sữa tươi tiệt trùng'\n"
                    "   - 'Bánh quy socola'\n"
                    "   - 'Phô mai con bò cười'"
                )

            if self.megallm_ready:
                logger.info(" Đang phân tích với AI...")
                context = self.format_context_for_megallm(results)
                response = self.ask_megallm(context, question_str)
                
                product_section = "\n\n SẢN PHẨM LIÊN QUAN:\n"
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    product_id = self.get_product_id(metadata)
                    product_name = metadata.get('name', 'Sản phẩm')
                    
                    if product_id and product_id != "unknown":
                        product_link = self.generate_product_link(product_id)
                        product_section += f"• {product_name} (ID: {product_id})\n"
                        product_section += f"   Link: {product_link}\n"
                    else:
                        product_section += f"• {product_name}\n"
                
                response += product_section
                logger.info(" Đã tạo câu trả lời với AI")
                return response
            else:
                logger.info(" Sử dụng chế độ cơ bản...")
                return self.smart_fallback_response(results, question_str)
                
        except Exception as e:
            logger.error(f" Lỗi hệ thống: {e}", exc_info=True)
            return " Xin lỗi, tôi gặp sự cố kỹ thuật. Vui lòng thử lại sau."

def main():
    rag_system = ChromaRAGSystem()
    
    if not rag_system.collection:
        print(" Không thể khởi tạo hệ thống. Thoát...")
        return

    print("\n" + "=" * 60)
    print("   🤖 HỆ THỐNG TƯ VẤN THỰC PHẨM THÔNG MINH")
    print("=" * 60)

    if rag_system.megallm_ready:
        print(" Đang sử dụng AI MegaLLM để phân tích nâng cao")
    else:
        print("ℹ Chế độ cơ bản (vẫn tìm kiếm được sản phẩm)")

    print("\n💡 Tôi có thể giúp bạn về thực phẩm và đồ uống:")
    print("   • Thông tin sản phẩm cụ thể")
    print("   • Thành phần và công dụng") 
    print("   • So sánh giá cả")
    print("   • Gợi ý sản phẩm liên quan")
    print("   • Cung cấp link chi tiết sản phẩm")

    print("\n📝 Nhập 'thoát' để kết thúc")
    print("=" * 60)

    while True:
        user_input = input("\n Bạn muốn tìm gì?: ").strip()
        
        if user_input.lower() in ['thoát', 'exit', 'quit', 'q', 'stop']:
            print("\n Cảm ơn bạn đã sử dụng dịch vụ! Hẹn gặp lại!")
            break
            
        if not user_input:
            continue

        print("🔄 Đang xử lý...")
        
        answer = rag_system.get_answer(user_input)
        print(f"\n{answer}")

chat_bot = None

def get_chat_bot():
    global chat_bot
    if chat_bot is None:
        chat_bot = ChromaRAGSystem()
    return chat_bot

if __name__ == "__main__":
    main()