from dotenv import load_dotenv
load_dotenv(override=True)

from pipelines.rag_pipeline import run

print(run("What is self-attention?", top_k=20, top_n=5))
