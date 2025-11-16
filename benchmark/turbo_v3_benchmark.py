import numpy as np
import time
import multiprocessing as mp
from pymilvus import connections, Collection

# ============================================================
# TURBO V3 (FIX gRPC + Multiprocessing)
# ============================================================
HOST = "127.0.0.1"
PORT = "19530"
COL_NAME = "faces_1m"

DIM = 512
TOTAL_QUERY = 5000
BATCH_SIZE = 8
PROCESS_COUNT = 8
TOPK = 5

# ============================================================
# FORCE SPAWN (IMPORTANT!!!)
# ============================================================
mp.set_start_method("spawn", force=True)

# ============================================================
# PRE-GENERATE QUERIES (shared memory safe)
# ============================================================
all_queries = np.random.rand(TOTAL_QUERY, DIM).astype("float32")


# ============================================================
# WORKER - RUN ONLY IN CHILD PROCESS
# ============================================================
def worker(batch_index):
    # CONNECT AGAIN (safe)
    connections.connect("default", host=HOST, port=PORT)
    c = Collection(COL_NAME)
    c.load()

    params = {"metric_type": "L2", "params": {"nprobe": 4}}

    s = batch_index * BATCH_SIZE
    e = s + BATCH_SIZE
    batch = all_queries[s:e]

    t0 = time.time()
    c.search(batch, "embedding", params, limit=TOPK)
    t1 = time.time()
    return t1 - t0


# ============================================================
# MAIN BENCH
# ============================================================
def run_bench():
    num_batches = TOTAL_QUERY // BATCH_SIZE

    print("========= TURBO V3 BENCHMARK =========")
    print(f"Total Query:     {TOTAL_QUERY}")
    print(f"Batch size:      {BATCH_SIZE}")
    print(f"Processes:       {PROCESS_COUNT}")
    print(f"Total batches:   {num_batches}")
    print("======================================")

    start_total = time.time()

    with mp.Pool(PROCESS_COUNT) as pool:
        lat_list = pool.map(worker, range(num_batches))

    end_total = time.time()

    lat_ms = np.array(lat_list) * 1000

    print("\n========= RESULTS (TURBO V3) =========")
    print(f"Total Time:     {end_total - start_total:.2f} sec")
    print(f"QPS:            {TOTAL_QUERY / (end_total - start_total):.2f} q/s")
    print(f"Avg Latency:    {lat_ms.mean():.2f} ms")
    print(f"P50 Latency:    {np.percentile(lat_ms, 50):.2f} ms")
    print(f"P95 Latency:    {np.percentile(lat_ms, 95):.2f} ms")
    print(f"P99 Latency:    {np.percentile(lat_ms, 99):.2f} ms")
    print("======================================")

if __name__ == "__main__":
    run_bench()
