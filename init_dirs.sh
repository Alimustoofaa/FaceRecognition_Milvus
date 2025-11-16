#!/usr/bin/env bash
set -e

BASE="./data/milvus-standalone"

echo "📂 Creating Milvus Standalone directory structure at: $BASE"

mkdir -p \
  "$BASE/etcd" \
  "$BASE/minio" \
  "$BASE/milvus/db" \
  "$BASE/milvus/wal" \
  "$BASE/milvus/logs" \
  "$BASE/milvus/cache" \
  "$BASE/milvus/object" \
  "$BASE/milvus/analyzer" \
  "$BASE/milvus/config" 

chmod -R 777 $BASE

cp milvus.yaml "$BASE/milvus/config/milvus.yaml"
echo "📄 milvus.yaml copied to $BASE/milvus/config/"

echo "✅ Done. Directory structure created:"
tree $BASE || find $BASE -maxdepth 4 -type d

echo "You can now run docker-compose."
