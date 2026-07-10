#!/bin/bash
# Setup Elasticsearch 7.x for BM25 first-stage retrieval
# The BEIR library's BM25Search requires a local Elasticsearch instance.
# Run this ONCE on your cluster node before any experiments.

set -e

ES_VERSION="7.17.9"

echo "=== Checking if Elasticsearch is already running ==="
if curl -s http://localhost:9200 > /dev/null 2>&1; then
    echo "Elasticsearch is already running at localhost:9200"
    curl -s http://localhost:9200
    exit 0
fi

echo "=== Checking if Elasticsearch is installed ==="
if [ -d "$HOME/elasticsearch-${ES_VERSION}" ]; then
    echo "Found existing installation at $HOME/elasticsearch-${ES_VERSION}"
else
    echo "=== Downloading Elasticsearch ${ES_VERSION} ==="
    cd "$HOME"
    wget -q "https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-${ES_VERSION}-linux-x86_64.tar.gz"
    tar xzf "elasticsearch-${ES_VERSION}-linux-x86_64.tar.gz"
    rm "elasticsearch-${ES_VERSION}-linux-x86_64.tar.gz"
    echo "Installed to $HOME/elasticsearch-${ES_VERSION}"
fi

echo "=== Starting Elasticsearch ==="
cd "$HOME/elasticsearch-${ES_VERSION}"

# Single-node discovery (no cluster needed)
export ES_JAVA_OPTS="-Xms8g -Xmx8g"
./bin/elasticsearch -d -p pid \
    -E discovery.type=single-node \
    -E xpack.security.enabled=false

echo "Waiting for Elasticsearch to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:9200 > /dev/null 2>&1; then
        echo "Elasticsearch is ready!"
        curl -s http://localhost:9200
        exit 0
    fi
    sleep 2
done

echo "ERROR: Elasticsearch did not start within 60 seconds"
exit 1
