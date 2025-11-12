import json
import re

def clean_text(text):
    """Làm sạch văn bản: chuẩn hóa dấu, khoảng trắng"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    if text and not text.endswith('.'):
        text += '.'
    return text

def product_to_chunk(product):
    """Chuyển 1 sản phẩm JSON thành 1 đoạn văn mô tả đầy đủ cho RAG"""
    parts = []

    name = product.get("name", "").strip()
    if not name:
        return None

    # Phần mở đầu
    parts.append(f'Sản phẩm "{name}"')

    # Mô tả (nếu có thông tin thực tế)
    desc = product.get("description", "")
    if desc and "Cung cấp dinh dưỡng cơ bản" not in desc:
        # Nếu mô tả custom → dùng
        parts[-1] += f" - {desc}"
    else:
        # Nếu mô tả chung → bỏ, dùng thông tin chi tiết bên dưới
        pass

    # Thành phần (quan trọng)
    ing = product.get("ingredients", "").strip()
    if ing and "nguyên liệu tự nhiên theo nhãn" not in ing.lower():
        parts.append(f"Thành phần: {ing}.")

    # Công dụng/lợi ích
    ben = product.get("benefits", "").strip()
    if ben and "Cung cấp dinh dưỡng cơ bản" not in ben:
        parts.append(f"Lợi ích: {ben}.")

    # Hướng dẫn bảo quản
    sto = product.get("storage", "").strip()
    if sto and "theo hướng dẫn trên bao bì" not in sto.lower():
        parts.append(f"Hướng dẫn bảo quản: {sto}.")

    # Dị nguyên
    aller = product.get("allergens", "").strip()
    if aller and aller.lower() not in ["có thể chứa: sữa, gluten, đậu nành (tuỳ sản phẩm)", "có thể chứa: sữa, gluten, đậu nành (tuỳ loại)"]:
        parts.append(f"Dị nguyên: {aller}.")

    # Đối tượng sử dụng
    target = product.get("target_audience", "").strip()
    if target and "người trưởng thành, tuỳ theo chỉ dẫn trên nhãn" not in target.lower():
        parts.append(f"Phù hợp cho: {target}.")

    # Ghép lại
    full_text = " ".join(parts)
    full_text = clean_text(full_text)
    return full_text

def prepare_rag_data(input_json_path, output_chunks_path=None):
    with open(input_json_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    chunks = []
    metadatas = []

    for prod in products:
        # Bỏ qua nếu ingredients chung chung và không có thông tin chi tiết nào
        ing = prod.get("ingredients", "")
        if "nguyên liệu tự nhiên theo nhãn" in ing.lower():
            # Kiểm tra thêm: nếu không có thông tin nào hữu ích → bỏ
            useful_fields = [
                prod.get("benefits", ""),
                prod.get("storage", ""),
                prod.get("allergens", ""),
                prod.get("target_audience", "")
            ]
            has_useful = any(
                field and "cơ bản" not in field.lower() and "theo hướng dẫn" not in field.lower()
                for field in useful_fields
            )
            if not has_useful:
                continue

        chunk = product_to_chunk(prod)
        if chunk and len(chunk) > 30:  # lọc đoạn quá ngắn
            chunks.append(chunk)
            metadatas.append({
                "product_id": prod.get("product_id", ""),
                "name": prod.get("name", ""),
                "category": prod.get("category", ""),
                "unit": prod.get("unit", ""),
                "price": prod.get("price", 0)
            })

    print(f"✅ Đã chuẩn bị {len(chunks)} chunks từ {len(products)} sản phẩm.")

    # Lưu (tuỳ chọn)
    if output_chunks_path:
        output = [{"text": t, "metadata": m} for t, m in zip(chunks, metadatas)]
        with open(output_chunks_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"💾 Đã lưu dữ liệu RAG vào: {output_chunks_path}")

    return chunks, metadatas

# === SỬ DỤNG ===
if __name__ == "__main__":
    # Lưu file JSON của bạn thành "products.json" trong cùng thư mục
    chunks, metadatas = prepare_rag_data(
        input_json_path="products.json",
        output_chunks_path="rag_chunks.json"
    )

    # In mẫu 2 chunks đầu tiên để kiểm tra
    for i in range(min(2, len(chunks))):
        print(f"\n--- Chunk {i+1} ---")
        print("Text:", chunks[i])
        print("Metadata:", metadatas[i])