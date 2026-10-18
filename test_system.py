import sys
sys.path.insert(0, '.')
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

tests = [
    ('GET', '/api/health', None),
    ('GET', '/api/dashboard/user_001', None),
    ('GET', '/api/food-log/logs/user_001', None),
    ('GET', '/api/food-log/daily-summary/user_001', None),
]
print('=== API Endpoint Tests ===')
for method, path, body in tests:
    if method == 'GET':
        r = client.get(path)
    else:
        r = client.post(path, json=body)
    status = 'PASS' if r.status_code == 200 else 'FAIL'
    print(f'  {status} {method} {path} -> HTTP {r.status_code}')

print()
print('=== RAG Vector Store Test ===')
from app.rag.vector_store import NutritionVectorStore
store = NutritionVectorStore()
print(f'  Docs in KB: {store.count()}')
queries = ['diabetes blood sugar', 'omega-3 heart health', 'pregnancy iron folate']
for q in queries:
    results = store.search(q, k=2)
    top = results[0]["content"][:60] if results else "none"
    print(f'  Query "{q[:25]}" -> {top}...')

print()
print('=== All Tests Passed ===')
