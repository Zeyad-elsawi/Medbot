"""Quick smoke test — verifies all API endpoints are reachable."""
import json
import sys
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"

# 1. Root
r = urllib.request.urlopen(f"{BASE}/")
print(f"GET /              -> {r.status} OK")

# 2. Swagger UI
r = urllib.request.urlopen(f"{BASE}/docs")
print(f"GET /docs          -> {r.status} OK")

# 3. Health — most important check
r = urllib.request.urlopen(f"{BASE}/api/health")
data = json.loads(r.read())
assert data["status"] in ("ok", "degraded"), f"Unexpected status: {data}"
assert data["qdrant_connected"] is True, "FAIL: Qdrant NOT connected!"
print(f"GET /api/health    -> {r.status} OK")
print(f"  status           : {data['status']}")
print(f"  qdrant_connected : {data['qdrant_connected']}")
print(f"  models_loaded    : {data['models_loaded']}")
print(f"  environment      : {data['environment']}")

# 4. PDF 404 — expected for nonexistent report
try:
    urllib.request.urlopen(f"{BASE}/api/report/fake-id-123/pdf")
    print("FAIL: Expected 404 but got 200")
    sys.exit(1)
except urllib.error.HTTPError as e:
    assert e.code == 404, f"Expected 404, got {e.code}"
    print(f"GET /api/report/fake/pdf -> 404 OK (expected)")

print("\nAll smoke tests PASSED.")
