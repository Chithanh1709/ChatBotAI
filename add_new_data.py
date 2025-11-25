import chromadb
import uuid

# KẾT NỐI VÀO CÙNG SERVER
client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.get_or_create_collection(name="kho_hang")

print("--- CÔNG CỤ THÊM SẢN PHẨM ---")

while True:
    ten_sp = input("\nNhập mô tả sản phẩm (hoặc 'q' để thoát): ")
    if ten_sp.lower() == 'q': break
    
    gia_sp = input("Nhập giá: ")

    # Gửi lệnh thêm vào Server
    collection.add(
        documents=[ten_sp],
        metadatas=[{"gia": gia_sp}],
        ids=[str(uuid.uuid4())]
    )
    
    print(f"✅ Đã thêm '{ten_sp}'! (Qua bên Bot hỏi thử xem)")