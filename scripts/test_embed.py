from dotenv import load_dotenv
load_dotenv(override=True)

from core.embedder import embed_query

q = "What is self-attention?"
vec = embed_query(q)
print("len =", len(vec))
print("first5 =", vec[:5])
