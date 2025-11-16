from pymilvus import *
import numpy as np

connections.connect("default", host="127.0.0.1", port="19530")
col = Collection("faces_1m")

page_size = 10000
total = col.num_entities
pages = (total // page_size) + 1

all_ids = []
all_vecs = []

for p in range(pages):
    offset = p * page_size

    res = col.query(
        expr=f"id >= {offset} && id < {offset + page_size}",
        output_fields=["id", "embedding"]
    )

    for r in res:
        all_ids.append(r["id"])
        all_vecs.append(r["embedding"])

    print(f"Page {p+1}/{pages} downloaded")

all_ids = np.array(all_ids)
all_vecs = np.array(all_vecs, dtype="float32")

np.save("ids.npy", all_ids)
np.save("vecs.npy", all_vecs)

print("Export completed!")
