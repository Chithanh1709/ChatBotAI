import chromadb
import time

# KẾT NỐI VÀO SERVER (localhost:8000)
# Thay vì đọc file trực tiếp, nó sẽ hỏi Server
try:
    client = chromadb.HttpClient(host='localhost', port=8000)
    collection = client.get_or_create_collection(name="food_products_vn")
    print("✅ Đã kết nối tới ChromaDB Server!")
except Exception as e:
    print("❌ Không tìm thấy Server! Bạn đã chạy lệnh 'chroma run' chưa?")
    exit()

print("--- BOT ĐANG CHẠY (Sẵn sàng nhận dữ liệu mới realtime) ---")

while True:
    query = input("\n🤖 Bạn muốn tìm gì? (gõ 'q' để thoát): ")
    if query.lower() == 'q': break

    # Gửi câu hỏi lên Server
    results = collection.query(
        query_texts=[query],
        n_results=1 
    )

    # Server trả về kết quả mới nhất
    if results['documents'] and results['documents'][0]:
        doc = results['documents'][0][0]
        meta = results['metadatas'][0][0]
        print(f"👉 Tìm thấy: {doc}")
        print(f"   (Chi tiết: {meta})")
    else:
        print("📭 Chưa tìm thấy sản phẩm nào phù hợp.")