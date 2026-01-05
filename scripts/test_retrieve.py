from dotenv import load_dotenv
load_dotenv(override=True)

from core.embedder import embed_query
from core.search import vector_search

q = "What is self-attention?"
vec = embed_query(q)

docs = vector_search(vec, top_k=5)
print("retrieved:", len(docs))

for i, d in enumerate(docs, 1):
    print("\n#", i, "page", d.get("page"))
    print(d.get("source_original"))
    txt = (d.get("content") or "").replace("\n", " ")
    print(txt[:250])
