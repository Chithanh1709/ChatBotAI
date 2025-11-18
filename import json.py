import json

# Đọc file JSON
with open('rag_chunks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Cập nhật product_id bắt đầu từ 3
for index, item in enumerate(data, start=4):
    item['metadata']['product_id'] = str(index)

# Ghi lại file JSON
with open('rag_chunks.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Đã cập nhật {len(data)} sản phẩm với product_id từ 3 đến {len(data) + 2}")