from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection
)

HOST = "127.0.0.1"
PORT = "19530"
COLLECTION_NAME = "test_face"

print("🔌 Connect to Milvus...")
connections.connect(alias="default", host=HOST, port=PORT)

# Drop kalau sudah ada
if COLLECTION_NAME in Collection.list():
    Collection(COLLECTION_NAME).drop()

print("📐 Define schema...")
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
]
schema = CollectionSchema(fields, description="test face embedding")

print("🧱 Create collection...")
collection = Collection(name=COLLECTION_NAME, schema=schema)

print("📥 Insert dummy data...")
import random
import numpy as np

vectors = np.random.rand(10, 512).astype("float32").tolist()
entities = [
    vectors,
]
# karena auto_id=True, ga perlu kirim id
collection.insert([vectors])

print("🔍 Build index (IVF_FLAT simple)...")
index_params = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024},
}
collection.create_index(field_name="embedding", index_params=index_params)

print("📦 Load collection...")
collection.load()

print("🔎 Search test...")
query_vec = np.random.rand(1, 512).astype("float32").tolist()
res = collection.search(
    data=query_vec,
    anns_field="embedding",
    param={"metric_type": "IP", "params": {"nprobe": 16}},
    limit=5,
)

for hits in res:
    for h in hits:
        print(f"ID={h.id}, distance={h.distance}")

print("✅ Done.")
