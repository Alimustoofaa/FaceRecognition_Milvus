# 🚀 Milvus Standalone Deployment (Optimized for 300M Face Embedding)

Milvus Standalone + MinIO + Etcd untuk skala **1M–300M face embedding**, dengan performa:

- **QPS**: ~2400 query/second  
- **Avg Latency**: ~3.7 ms  
- **P99 Latency**: < 10 ms  
- GPU-accelerated ANN search  
- Cocok untuk CCTV, gate access, mobile FR, edge & datacenter

---

## 📦 Features

- Milvus Standalone v2.6.5-GPU  
- MinIO object storage  
- Etcd metadata store  
- RocksMQ (default, simple & stable)  
- Optimized for IVF / HNSW / DiskANN  
- Multiprocessing-safe benchmark (TURBO V3)  
- Real-time latency < 10ms

---

## 📂 Directory Structure

```
milvus_deployment/
 ├── docker-compose.yml
 ├── init_dirs.sh
 ├── README.md
 └── benchmark/
       └── turbo_v3_benchmark.py
```

---

## 🔌 Port Mapping

| Service            | Host Port | Container Port | Description                      |
|--------------------|-----------|----------------|----------------------------------|
| Milvus gRPC        | 19530     | 19530          | PyMilvus / Client API            |
| Milvus REST / Web  | 9091      | 9091           | Web UI, REST, metrics            |
| MinIO S3 API       | 9000      | 9000           | S3-compatible storage endpoint   |
| MinIO Console      | 9001      | 9001           | Web UI MinIO                     |
| etcd Client (int)  | —         | 2379           | Internal Milvus metadata         |
| etcd Peer (int)    | —         | 2380           | Internal etcd cluster comm       |

---

## ⚙️ Installation & Setup

### 1️⃣ Generate directories

```bash
sudo chmod +x init_dirs.sh
sudo ./init_dirs.sh
```

Resulting structure:

```
/data/milvus-standalone/
 ├── etcd/
 ├── minio/
 └── milvus/
      ├── db/
      ├── wal/
      ├── logs/
      ├── cache/
      ├── object/
      └── analyzer/
```

---

### 2️⃣ Start services

```bash
docker compose up -d
```

---

### 3️⃣ Check status

```bash
docker ps
```

Semua harus status **Up**:

- milvus-etcd  
- milvus-minio  
- milvus-standalone  

---

### 4️⃣ Check Milvus logs

```bash
docker logs -f milvus-standalone
```

Harus muncul:

```
Milvus standalone is ready
```

---

## 🧪 Benchmark TURBO V3 (Multiprocessing Safe)

Jalankan:

```bash
python3 benchmark/turbo_v3_benchmark.py
```

Contoh hasil nyata:

```
QPS:            2393.45 q/s
Avg Latency:    3.72 ms
P95 Latency:    5.00 ms
P99 Latency:    9.69 ms
```

---
