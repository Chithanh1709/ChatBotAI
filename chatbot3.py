import os, requests, json, logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("RAG")

MEGA_URL = "https://ai.megallm.io/v1/chat/completions"
MEGA_KEY = "sk-mega-9e02941cc7286047dfe1dc53d2d94a1afddddd677e4769b5189ed82a992f2f43"       
MEGA_MODEL = "llama3.3-70b-instruct"

DB_PATH = "D:/chroma_food_rag"
COL = "food_products_vn"
PRODUCT_URL = "http://localhost:4200/product"

emb = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert", model_kwargs={"device":"cpu"})
vecdb = Chroma(persist_directory=DB_PATH, embedding_function=emb, collection_name=COL)

def mega(messages, max_tokens=600, temperature=0.1):
    r = requests.post(
        MEGA_URL,
        headers={"Authorization":f"Bearer {MEGA_KEY}","Content-Type":"application/json"},
        json={"model":MEGA_MODEL,"messages":messages,"max_tokens":max_tokens,"temperature":temperature},
        timeout=60
    )
    if r.status_code!=200: return f"Lỗi MegaLLM: {r.text}"
    return r.json()["choices"][0]["message"]["content"]

def classify(query):
    msg=[
        {"role":"system","content":"Trả lời duy nhất 'FOOD' hoặc 'OTHER'. Nếu câu hỏi liên quan thực phẩm, sản phẩm ăn uống, thành phần, dinh dưỡng → FOOD."},
        {"role":"user","content":query}
    ]
    r = mega(msg, max_tokens=2).strip().upper()
    return r=="FOOD"

def ctx_format(docs):
    if not docs: return "Không có sản phẩm."
    out="DANH SÁCH SẢN PHẨM:\n\n"
    for i,d in enumerate(docs,1):
        m=d.metadata
        pid=m.get("product_id") or m.get("id") or "unknown"
        out+=f"=== SP {i} – ID {pid} ===\n"
        out+=f"Tên: {m.get('name','(không tên)')}\n"
        out+=f"Giá: {m.get('price','?')}\n"
        out+=f"Loại: {m.get('category','?')}\n"
        out+=f"{d.page_content}\n\n"
    return out

def rag_answer(query):
    docs = vecdb.similarity_search(query, k=4)
    if not docs: return "Không tìm thấy sản phẩm phù hợp."
    ctx = ctx_format(docs)
    prompt=[
        {"role":"system","content":"Bạn là chuyên gia tư vấn thực phẩm. Chỉ trả lời dựa trên context."},
        {"role":"user","content":f"THÔNG TIN:\n{ctx}\nCÂU HỎI: {query}\nTrả lời:"}
    ]
    ans = mega(prompt)
    ans+="\n\n🔗 SẢN PHẨM LIÊN QUAN:\n"
    for d in docs:
        m=d.metadata
        pid=m.get("product_id") or m.get("id") or None
        name=m.get("name","Sản phẩm")
        if pid:
            ans+=f"- {name}: {PRODUCT_URL}/{pid}\n"
        else:
            ans+=f"- {name}: (không có ID)\n"
    return ans

def chat(query):
    if not query.strip(): return "❗ Hãy nhập câu hỏi."
    if not classify(query): return "⚠️ Tôi chỉ tư vấn liên quan đến thực phẩm."
    return rag_answer(query)

if __name__=="__main__":
    print("=== RAG + MegaLLM 🥗 ===")
    while True:
        q=input("❓ ")
        if q.lower() in ["exit","quit","thoát"]: break
        print("👉", chat(q))