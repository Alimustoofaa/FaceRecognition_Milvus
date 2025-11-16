from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
import time

# ============================================================
# 1. Connect to Milvus GPU
# ============================================================
connections.connect("default", host="127.0.0.1", port="19530")
print("Connected to Milvus!")

# ============================================================
# 2. Create Collection (dim 512)
# ============================================================
dim = 512
col_name = "faces_1m"

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields, description="Face Recognition 1M Test")

# FIXED: milvus v2.x way to list collections
if col_name in utility.list_collections():
    print("Dropping existing collection:", col_name)
    Collection(col_name).drop()

collection = Collection(col_name, schema)
print(f"Collection `{col_name}` created successfully!")

# ============================================================
# 3. Insert 1 million embeddings (batched)
# ============================================================
N = 1_000_000
batch_size = 20_000     # aman untuk RAM
num_batches = N // batch_size

print(f"Inserting {N:,} vectors in {num_batches} batches...")

start_insert = time.time()

id_counter = 1

for i in range(num_batches):
    print(f"Batch {i+1}/{num_batches} inserting...")

    ids = np.arange(id_counter, id_counter + batch_size)
    vectors = np.random.rand(batch_size, dim).astype("float32")

    collection.insert([ids, vectors])
    id_counter += batch_size

collection.flush()

print(f"Insert completed in {time.time() - start_insert:.2f} seconds")

# ============================================================
# 4. Create IVF_FLAT Index
# ============================================================
print("Creating IVF_FLAT index on GPU...")

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 16384}   # terbaik untuk 1 juta
}

start_index = time.time()
collection.create_index("embedding", index_params)
print(f"Index created in {time.time() - start_index:.2f} seconds")

# ============================================================
# 5. Load to GPU
# ============================================================
print("Loading collection to GPU...")
start_load = time.time()
collection.load()   # GPU accelerated
print(f"Loaded in {time.time() - start_load:.2f} seconds")

# ============================================================
# 6. Test Search
# ============================================================
query = np.random.rand(1, dim).astype("float32")

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 64}
}

start_search = time.time()
results = collection.search(
    data=query,
    anns_field="embedding",
    param=search_params,
    limit=5,
    output_fields=["id"]
)
end_search = time.time()

print("Search result IDs:", results[0].ids)
print(f"Search time: {(end_search - start_search)*1000:.3f} ms")
