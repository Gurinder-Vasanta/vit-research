import chromadb
from chromadb import PersistentClient

client = PersistentClient(path="./chroma_store")

collection = client.get_or_create_collection(name="temp",metadata={"hnsw:space": "l2"})

embeddings = [
    [0.12312,0.74801892849,0.4718234781],
    [0.659128242,0.7583183479283,0.74921384027],
    [0.571940834820,0.735948247980,0.142780539123824]
]

ids=['c1','c2','c3']

metadatas = [
    {'label': 'vec1'},
    {'label': 'vec2'},
    {'label': 'vec3'}
]

collection.add(
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas
)

results = collection.query(
    query_embeddings=[[0.2, 0.25, 0.35]],
    n_results=2
)

print(results)

# load it back: 
# client = chromadb.Client(Settings(
#     chroma_db_impl="duckdb+parquet",
#     persist_directory="./chroma_store"
# ))