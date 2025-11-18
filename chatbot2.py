import os
from typing import List
import logging
import requests
import json

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Đang khởi tạo hệ thống RAG ...")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    print("Vui lòng cài: pip install langchain-huggingface")
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    print("Vui lòng cài: pip install langchain-chroma")
    from langchain_community.vectorstores import Chroma


# === CẤU HÌNH MEGA LLM ===
MEGALLM_BASE_URL = "https://ai.megallm.io/v1"
MEGALLM_API_KEY = "sk-mega-9e02941cc7286047dfe1dc53d2d94a1afddddd677e4769b5189ed82a992f2f43"  
MEGALLM_MODEL_NAME = "gpt-4"  

# === CẤU HÌNH KHÁC ===
CHROMA_DB_PATH = "D:/chroma_food_rag"
COLLECTION_NAME = "food_products_vn"
PRODUCT_BASE_URL = "http://localhost:4200/product"

def initialize_megallm_client():
    """Cấu hình MegaLLM client"""
    try:
        # Kiểm tra kết nối đến MegaLLM
        headers = {
            "Authorization": f"Bearer {MEGALLM_API_KEY}",
            "Content-Type": "application/json"
        }
        
        test_payload = {
            "model": MEGALLM_MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Xin chào"}
            ],
            "max_tokens": 50
        }
        
        response = requests.post(
            f"{MEGALLM_BASE_URL}/chat/completions", 
            json=test_payload, 
            headers=headers, 
            timeout=30
        )
        
        if response.status_code == 200:
            print("Đã kết nối thành công với MegaLLM")
            return True
        else:
            print(f"Lỗi kết nối MegaLLM: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Lỗi cấu hình MegaLLM: {e}")
        return False


def initialize_rag_system():
    """Khởi tạo hệ thống RAG"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="keepitreal/vietnamese-sbert",
            model_kwargs={"device": "cpu"}
        )
        print("Đã tải embedding model")

        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        print("Đã kết nối Chroma DB")

        test_results = vector_store.similarity_search("sữa", k=1)
        print(f"Test thành công. Tìm thấy {len(test_results)} documents")

        return vector_store
    except Exception as e:
        print(f"Lỗi khởi tạo RAG: {e}")
        return None


def create_megallm_prompt(context: str, question: str) -> List[dict]:
    """Tạo prompt cho MegaLLM theo format messages"""
    system_message = """Bạn là chuyên gia tư vấn thực phẩm. Hãy sử dụng thông tin sản phẩm được cung cấp để trả lời câu hỏi.

HƯỚNG DẪN:
- CHỈ sử dụng thông tin được cung cấp trong THÔNG TIN SẢN PHẨM
- KHÔNG bịa thêm thông tin
- Nếu không có thông tin phù hợp, nói "Không tìm thấy thông tin phù hợp"
- Trả lời bằng tiếng Việt
- Tập trung vào thông tin thực tế
- Khi đề cập đến sản phẩm, có thể tham khảo ID sản phẩm"""

    user_content = f"""THÔNG TIN SẢN PHẨM:
{context}

CÂU HỎI: {question}

Hãy trả lời dựa trên thông tin sản phẩm trên:"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]


def get_product_id(metadata: dict) -> str:
    """Lấy ID sản phẩm từ metadata (thử nhiều trường khác nhau)"""
    # Thử các trường ID có thể có
    possible_id_fields = ["id", "product_id", "productId", "ID", "productID"]
    
    for field in possible_id_fields:
        product_id = metadata.get(field)
        if product_id and product_id != "unknown":
            return str(product_id)
    
    return "unknown"


def format_context_for_megallm(docs: List) -> str:
    if not docs:
        return "Không có thông tin sản phẩm."

    context = "DANH SÁCH SẢN PHẨM:\n\n"
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        product_id = get_product_id(metadata)
        context += f"=== SẢN PHẨM {i} (ID: {product_id}) ===\n"
        context += f"Tên: {metadata.get('name', 'Chưa có tên')}\n"
        if metadata.get("price"):
            context += f"Giá: {metadata['price']:,} VNĐ\n"
        if metadata.get("category"):
            context += f"Danh mục: {metadata['category']}\n"
        if metadata.get("ingredients"):
            context += f"Thành phần: {metadata['ingredients']}\n"
        if metadata.get("benefits"):
            context += f"Lợi ích: {metadata['benefits']}\n"
        if metadata.get("storage"):
            context += f"Bảo quản: {metadata['storage']}\n"
        context += f"Mô tả: {doc.page_content}\n\n"
    return context


def ask_megallm(context: str, question: str) -> str:
    """Gọi MegaLLM API theo format từ documentation"""
    try:
        messages = create_megallm_prompt(context, question)
        
        payload = {
            "model": MEGALLM_MODEL_NAME,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.3
        }
        
        headers = {
            "Authorization": f"Bearer {MEGALLM_API_KEY}",
            "Content-Type": "application/json"
        }
        
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
            return f"Lỗi MegaLLM API: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Lỗi kết nối MegaLLM: {str(e)}"


def generate_product_link(product_id: str) -> str:
    """Tạo link chi tiết sản phẩm"""
    return f"{PRODUCT_BASE_URL}/{product_id}"


def smart_fallback_response(docs: List, question: str) -> str:
    if not docs:
        return "Không tìm thấy sản phẩm phù hợp."

    response = f"Tìm thấy {len(docs)} sản phẩm liên quan:\n\n"
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        product_id = get_product_id(metadata)
        product_link = generate_product_link(product_id)
        
        response += f"{i}. {metadata.get('name', 'Sản phẩm')}\n"
        response += f"   Link: {product_link}\n"
        response += f"   ID: {product_id}\n"
        if metadata.get("price"):
            response += f"   Giá: {metadata['price']:,} VNĐ\n"
        if metadata.get("category"):
            response += f"   Loại: {metadata['category']}\n"
        if metadata.get("benefits"):
            short = metadata["benefits"][:60] + "..." if len(metadata["benefits"]) > 60 else metadata["benefits"]
            response += f"   {short}\n"
        response += "\n"

    return response


def is_food_related(query: str) -> bool:
    query = query.lower()
    food_keywords = [
        "thực phẩm", "sản phẩm", "ăn", "uống", "nấu", "món",
        "thịt", "cá", "rau", "sữa", "bánh", "mì", "gạo", "đậu",
        "thành phần", "bảo quản", "dị ứng", "giá", "công dụng", "lợi ích"
    ]
    
    # Yêu cầu ít nhất một từ khóa thực phẩm
    has_food_keyword = any(kw in query for kw in food_keywords)
    
    
    too_general = query.strip() in [
        "giá", "bán gì", "có gì", "sản phẩm", "cung cấp gì",
        "bán chạy", "nhiều nhất", "tổng hợp"
    ]
    
    return has_food_keyword and not too_general


def main():
    megallm_ready = initialize_megallm_client()
    vector_store = initialize_rag_system()
    if not vector_store:
        return

    print("\n" + "=" * 60)
    print("HỆ THỐNG TƯ VẤN THỰC PHẨM")
    print("=" * 60)

    if megallm_ready:
        print("Đang sử dụng MegaLLM")
    else:
        print("Chế độ cơ bản (cần cấu hình MegaLLM API)")
        print("Vui lòng đặt MEGALLM_API_KEY và MEGALLM_MODEL_NAME")

    print("\nNhập 'thoát' để kết thúc")
    print("=" * 60)

    while True:
        user_input = input("\nBạn hỏi: ").strip()
        if user_input.lower() in ["thoát", "exit", "quit", "q"]:
            print("Tạm biệt!")
            break
        if not user_input:
            continue
        if not is_food_related(user_input):
            print("Tôi chỉ hỗ trợ câu hỏi về thực phẩm.")
            continue

        print("Đang tìm kiếm...")
        docs = vector_store.similarity_search(user_input, k=3)

        if not docs:
            print("Không tìm thấy sản phẩm phù hợp.")
            continue

        if megallm_ready:
            context = format_context_for_megallm(docs)
            response = ask_megallm(context, user_input)
            print(f"\nTrả lời:\n{response}")
            
            # Hiển thị link sản phẩm sau câu trả lời của MegaLLM
            print("\nSẢN PHẨM LIÊN QUAN:")
            for doc in docs:
                metadata = doc.metadata
                product_id = get_product_id(metadata)
                if product_id and product_id != "unknown":
                    product_link = generate_product_link(product_id)
                    product_name = metadata.get('name', 'Sản phẩm')
                    print(f"- {product_name}: {product_link}")
                else:
                    product_name = metadata.get('name', 'Sản phẩm')
                    print(f"- {product_name}: Không có ID")
        else:
            response = smart_fallback_response(docs, user_input)
            print(f"\n{response}")


class ChatBot:
    def __init__(self):
        logger.info("Initializing ChatBot...")
        self.megallm_ready = initialize_megallm_client()
        self.vector_store = initialize_rag_system()
        logger.info("ChatBot initialized")

    def get_answer(self, question: str) -> str:
        try:
            # VALIDATE đầu vào
            if question is None:
                return "Câu hỏi không hợp lệ (None)."
                
            question_str = str(question).strip() if question else ""
            
            if not question_str:
                return "Vui lòng nhập câu hỏi."
                
            if question_str.lower() in ["none", "null", "undefined"]:
                return "Câu hỏi không hợp lệ."
                
            logger.info(f"Processing question: '{question_str}'")
            
            # Kiểm tra hệ thống RAG
            if not self.vector_store:
                return "Hệ thống đang được bảo trì. Vui lòng thử lại sau."

            # Kiểm tra liên quan đến thực phẩm
            if not is_food_related(question_str):
                return (
                    "Tôi chỉ hỗ trợ câu hỏi về thực phẩm.\n"
                    "Vui lòng hỏi về sản phẩm cụ thể như:\n"
                    "   - 'Sữa tươi này có tốt không?'\n"
                    "   - 'Thành phần của bánh gạo là gì?'\n"
                    "   - 'Cách bảo quản phô mai?'"
                )

            # Tìm kiếm trong database
            logger.info("Searching in vector database...")
            docs = self.vector_store.similarity_search(question_str, k=3)
            logger.info(f"Found {len(docs)} relevant documents")

            # Nếu không tìm thấy tài liệu phù hợp → yêu cầu cụ thể hơn
            if not docs:
                return (
                    "Không tìm thấy sản phẩm cụ thể phù hợp.\n"
                    "Vui lòng nêu rõ tên sản phẩm hoặc mô tả chi tiết hơn, ví dụ:\n"
                    "   - 'Thông tin về sữa TH True Milk?'\n"
                    "   - 'Bánh quy Oreo có đường không?'\n"
                    "   - 'Giá của phô mai Mozzarella bao nhiêu?'"
                )

            # Nếu có tài liệu → tiếp tục xử lý
            if self.megallm_ready:
                logger.info("Using MegaLLM for response...")
                context = format_context_for_megallm(docs)
                response = ask_megallm(context, question_str)
                
                # Thêm thông tin sản phẩm với link
                product_links = "\n\nSẢN PHẨM LIÊN QUAN:\n"
                for doc in docs:
                    metadata = doc.metadata
                    product_id = get_product_id(metadata)
                    if product_id and product_id != "unknown":
                        product_link = generate_product_link(product_id)
                        product_name = metadata.get('name', 'Sản phẩm')
                        product_links += f"- {product_name}: {product_link}\n"
                    else:
                        product_name = metadata.get('name', 'Sản phẩm')
                        product_links += f"- {product_name}: Không có ID\n"
                
                response += product_links
                logger.info("MegaLLM response generated with product links")
                return response
            else:
                logger.info("Using fallback response...")
                return smart_fallback_response(docs, question_str)
                
        except Exception as e:
            logger.error(f"Error in get_answer: {e}", exc_info=True)
            return f"Đã xảy ra lỗi hệ thống: {str(e)}"


if __name__ == "__main__":
    main()