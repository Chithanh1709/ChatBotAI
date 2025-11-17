import os
from typing import List
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(" Đang khởi tạo hệ thống RAG với Google Gemini...")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print(" Đã import google.generativeai")
except ImportError:
    print(" Không thể import Google Generative AI")
    GEMINI_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    print(" Vui lòng cài: pip install langchain-huggingface")
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    print(" Vui lòng cài: pip install langchain-chroma")
    from langchain_community.vectorstores import Chroma


# === CẤU HÌNH ===
CHROMA_DB_PATH = "D:/chroma_food_rag"
COLLECTION_NAME = "food_products_vn"
GEMINI_MODEL_NAME = "models/gemini-pro-latest"  
PRODUCT_BASE_URL = "http://localhost:4200/products"

def initialize_gemini_client():
    """Cấu hình API key cho Gemini (không dùng Client)"""
    if not GEMINI_AVAILABLE:
        return None

    try:
        genai.configure(api_key="AIzaSyBEM0RjTfvX1LW0IHZcqvZOo51s9TIlhSE")
        print(" Đã cấu hình Google Gemini")

        # Kiểm tra model có khả dụng không 
        try:
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            _ = model.generate_content("Xin chào")
            print(f" Model {GEMINI_MODEL_NAME} khả dụng")
        except Exception as e:
            print(f" Model {GEMINI_MODEL_NAME} có thể không khả dụng: {e}")

        return True
    except Exception as e:
        print(f" Lỗi cấu hình Gemini: {e}")
        return None


def initialize_rag_system():
    """Khởi tạo hệ thống RAG"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="keepitreal/vietnamese-sbert",
            model_kwargs={"device": "cpu"}
        )
        print(" Đã tải embedding model")

        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        print(" Đã kết nối Chroma DB")

        test_results = vector_store.similarity_search("sữa", k=1)
        print(f" Test thành công. Tìm thấy {len(test_results)} documents")

        return vector_store
    except Exception as e:
        print(f" Lỗi khởi tạo RAG: {e}")
        return None


def create_gemini_prompt(context: str, question: str) -> str:
    return f"""Bạn là chuyên gia tư vấn thực phẩm. Hãy sử dụng thông tin sản phẩm dưới đây để trả lời câu hỏi.

THÔNG TIN SẢN PHẨM:
{context}

CÂU HỎI: {question}

HƯỚNG DẪN:
- CHỈ sử dụng thông tin được cung cấp
- KHÔNG bịa thông tin
- Nếu không có thông tin, nói "Không tìm thấy"
- Trả lời bằng tiếng Việt
- Tập trung vào thông tin thực tế
- Khi đề cập đến sản phẩm, có thể tham khảo ID sản phẩm

Trả lời:"""


def format_context_for_gemini(docs: List) -> str:
    if not docs:
        return "Không có thông tin sản phẩm."

    context = "DANH SÁCH SẢN PHẨM:\n\n"
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        product_id = metadata.get("id", "unknown")
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


def ask_gemini(context: str, question: str) -> str:
    """Gọi Gemini đúng cách qua GenerativeModel"""
    try:
        prompt = create_gemini_prompt(context, question)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f" Lỗi Gemini: {str(e)}"


def generate_product_link(product_id: str) -> str:
    """Tạo link chi tiết sản phẩm"""
    return f"{PRODUCT_BASE_URL}/{product_id}"


def smart_fallback_response(docs: List, question: str) -> str:
    if not docs:
        return "Không tìm thấy sản phẩm phù hợp."

    response = f"🔍 Tìm thấy {len(docs)} sản phẩm liên quan:\n\n"
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        product_id = metadata.get("id", "unknown")
        product_link = generate_product_link(product_id)
        
        response += f"{i}. **{metadata.get('name', 'Sản phẩm')}**\n"
        response += f"   🔗 [Xem chi tiết]({product_link})\n"
        response += f"   📋 ID: {product_id}\n"
        if metadata.get("price"):
            response += f"   💰 Giá: {metadata['price']:,} VNĐ\n"
        if metadata.get("category"):
            response += f"   🏷️ Loại: {metadata['category']}\n"
        if metadata.get("benefits"):
            short = metadata["benefits"][:60] + "..." if len(metadata["benefits"]) > 60 else metadata["benefits"]
            response += f"   ✅ {short}\n"
        response += "\n"

    response += " *Để có câu trả lời chi tiết, hãy cấu hình Google Gemini API*"
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
    gemini_ready = initialize_gemini_client()
    vector_store = initialize_rag_system()
    if not vector_store:
        return

    print("\n" + "=" * 60)
    print(" HỆ THỐNG TƯ VẤN THỰC PHẨM")
    print("=" * 60)

    if gemini_ready:
        print(" Đang sử dụng Google Gemini")
    else:
        print("Chế độ cơ bản (cần cài google-generativeai)")
        print("Chạy: pip install -U google-generativeai")

    print("\nNhập 'thoát' để kết thúc")
    print("=" * 60)

    while True:
        user_input = input("\n Bạn hỏi: ").strip()
        if user_input.lower() in ["thoát", "exit", "quit", "q"]:
            print(" Tạm biệt!")
            break
        if not user_input:
            continue
        if not is_food_related(user_input):
            print(" Tôi chỉ hỗ trợ câu hỏi về thực phẩm.")
            continue

        print(" Đang tìm kiếm...")
        docs = vector_store.similarity_search(user_input, k=3)

        if not docs:
            print(" Không tìm thấy sản phẩm phù hợp.")
            continue

        if gemini_ready:
            context = format_context_for_gemini(docs)
            response = ask_gemini(context, user_input)
            print(f"\n Trả lời:\n{response}")
            
            # Hiển thị link sản phẩm sau câu trả lời của Gemini
            print("\n📋 SẢN PHẨM LIÊN QUAN:")
            for doc in docs:
                metadata = doc.metadata
                product_id = metadata.get("product_id")
                if product_id and product_id != "unknown":
                    product_link = generate_product_link(product_id)
                    print(f"• {metadata.get('name', 'Sản phẩm')}: {product_link}")
        else:
            response = smart_fallback_response(docs, user_input)
            print(f"\n{response}")

class ChatBot:
    def __init__(self):
        logger.info(" Initializing ChatBot...")
        self.gemini_ready = initialize_gemini_client()
        self.vector_store = initialize_rag_system()
        logger.info(" ChatBot initialized")

    def get_answer(self, question: str) -> str:
        try:
            # VALIDATE đầu vào
            if question is None:
                return " Câu hỏi không hợp lệ (None)."
                
            question_str = str(question).strip() if question else ""
            
            if not question_str:
                return " Vui lòng nhập câu hỏi."
                
            if question_str.lower() in ["none", "null", "undefined"]:
                return " Câu hỏi không hợp lệ."
                
            logger.info(f" Processing question: '{question_str}'")
            
            # Kiểm tra hệ thống RAG
            if not self.vector_store:
                return " Hệ thống đang được bảo trì. Vui lòng thử lại sau."

            # Kiểm tra liên quan đến thực phẩm
            if not is_food_related(question_str):
                return (
                    " Tôi chỉ hỗ trợ câu hỏi về thực phẩm.\n"
                    " Vui lòng hỏi về sản phẩm cụ thể như:\n"
                    "   • 'Sữa tươi này có tốt không?'\n"
                    "   • 'Thành phần của bánh gạo là gì?'\n"
                    "   • 'Cách bảo quản phô mai?'"
                )

            # Tìm kiếm trong database
            logger.info(" Searching in vector database...")
            docs = self.vector_store.similarity_search(question_str, k=3)
            logger.info(f" Found {len(docs)} relevant documents")

            # Nếu không tìm thấy tài liệu phù hợp → yêu cầu cụ thể hơn
            if not docs:
                return (
                    "🔍 Không tìm thấy sản phẩm cụ thể phù hợp.\n"
                    "👉 Vui lòng nêu rõ tên sản phẩm hoặc mô tả chi tiết hơn, ví dụ:\n"
                    "   • 'Thông tin về sữa TH True Milk?'\n"
                    "   • 'Bánh quy Oreo có đường không?'\n"
                    "   • 'Giá của phô mai Mozzarella bao nhiêu?'"
                )

            # Nếu có tài liệu → tiếp tục xử lý như cũ
            if self.gemini_ready:
                logger.info(" Using Gemini for response...")
                context = format_context_for_gemini(docs)
                response = ask_gemini(context, question_str)
                
                # Thêm thông tin sản phẩm với link
                product_links = "\n\n🔗 **SẢN PHẨM LIÊN QUAN:**\n"
                for doc in docs:
                    metadata = doc.metadata
                    product_id = metadata.get("id")
                    if product_id and product_id != "unknown":
                        product_link = generate_product_link(product_id)
                        product_links += f"• [{metadata.get('name', 'Sản phẩm')}]({product_link})\n"
                
                response += product_links
                logger.info(" Gemini response generated with product links")
                return response
            else:
                logger.info(" Using fallback response...")
                return smart_fallback_response(docs, question_str)
                
        except Exception as e:
            logger.error(f" Error in get_answer: {e}", exc_info=True)
            return f"❌ Đã xảy ra lỗi hệ thống: {str(e)}"


if __name__ == "__main__":
    main()