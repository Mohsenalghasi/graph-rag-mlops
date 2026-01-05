from dotenv import load_dotenv
load_dotenv(override=True)

import os
import requests
from openai import AzureOpenAI

q = "What is self-attention?"

# Azure OpenAI embedding
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

vec = client.embeddings.create(
    model=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"],
    input=q
).data[0].embedding

print("EMBED LEN:", len(vec))

# Azure Search vector query
ep = os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/")
key = os.environ["AZURE_SEARCH_API_KEY"]
idx = os.environ["AZURE_SEARCH_INDEX_NAME"]

url = f"{ep}/indexes/{idx}/docs/search?api-version=2024-07-01"

payload = {
    "vectorQueries": [
        {
            "kind": "vector",
            "vector": vec,
            "fields": "embedding",
            "k": 5
        }
    ],
    "top": 5,
    "select": "id,source_original,page,chunk_id"
}

r = requests.post(
    url,
    headers={
        "api-key": key,
        "Content-Type": "application/json"
    },
    json=payload,
    timeout=30
)

print("STATUS:", r.status_code)
print("BODY:", r.text[:1200])
