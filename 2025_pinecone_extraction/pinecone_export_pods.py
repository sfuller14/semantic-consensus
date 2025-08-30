# pinecone_export_pods.py
# pip install "pinecone[grpc]" tqdm pyyaml

import os, json, pathlib, time, math
from tqdm import tqdm
import yaml

# --- CONFIG (edit these) ---
API_KEY = os.environ.get("PINECONE_API_KEY") or "787ccc9e-d059-4004-b931-467f44861ac8"
HOST = "https://ecommerce-113856f.svc.us-east-1-aws.pinecone.io"
DIMENSION = 1536
NAMESPACES = {
    "reviews": 11750,
    "products": 908,
    "combined": 908,
}
ID_START_INCLUSIVE = 0          # looks like 0..n-1 in your setup
BATCH = 1000                    # safe batch size for fetch
OUTDIR = pathlib.Path("./pinecone_export")
RETRY_MAX = 5
# ---------------------------

from pinecone.grpc import PineconeGRPC as Pinecone

OUTDIR.mkdir(parents=True, exist_ok=True)
pc = Pinecone(api_key=API_KEY)
index = pc.Index(host=HOST)

def fetch_chunk(ids, ns):
    """Fetch with retries; return dict[id] -> {'values': [...], 'metadata': {...}}"""
    for attempt in range(1, RETRY_MAX + 1):
        try:
            res = index.fetch(ids=ids, namespace=ns)   # FetchResponse
            out = {}
            vecs = getattr(res, "vectors", {}) or {}
            for vid, vec in vecs.items():              # vec is a Vector object
                out[vid] = {
                    "values": list(getattr(vec, "values", []) or []),
                    "metadata": dict(getattr(vec, "metadata", {}) or {}),
                }
            return out
        except Exception as e:
            if attempt == RETRY_MAX:
                raise
            time.sleep(min(2 ** attempt, 10))

def export_namespace(ns, total_count):
    out_path = OUTDIR / f"export_{ns or 'default'}.jsonl"
    missing = 0
    found = 0

    with out_path.open("w", encoding="utf-8") as fout:
        # Sweep IDs [start, start+total_count)
        all_ids = [str(i) for i in range(ID_START_INCLUSIVE, ID_START_INCLUSIVE + total_count)]
        # Chunk
        for i in tqdm(range(0, len(all_ids), BATCH), desc=f"Exporting '{ns}'"):
            chunk = all_ids[i:i+BATCH]
            vectors = fetch_chunk(chunk, ns)
            # Write found vectors (fetch returns only those that exist)
            for vid, payload in vectors.items():
                rec = {
                    "id": vid,
                    "values": payload["values"],
                    "metadata": payload["metadata"],
                    "namespace": ns,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            found += len(vectors)
            missing += (len(chunk) - len(vectors))

    return {"written": found, "missing": missing, "outfile": str(out_path)}

def main():
    summary = {
        "host": HOST,
        "dimension": DIMENSION,
        "namespaces": {},
    }

    for ns, count in NAMESPACES.items():
        stats = export_namespace(ns, count)
        summary["namespaces"][ns] = stats

    # Write summary
    with open(OUTDIR / "export_summary.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("\nDone.")
    print("Summary:")
    print(yaml.safe_dump(summary, sort_keys=False))

if __name__ == "__main__":
    main()