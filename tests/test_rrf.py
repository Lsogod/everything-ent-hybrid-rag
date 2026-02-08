from app.services.search import reciprocal_rank_fusion


def test_rrf_merge_order():
    keyword_hits = [
        {"_id": "a", "_rank": 1, "content": "A"},
        {"_id": "b", "_rank": 2, "content": "B"},
    ]
    vector_hits = [
        {"_id": "b", "_rank": 1, "content": "B"},
        {"_id": "c", "_rank": 2, "content": "C"},
    ]
    merged = reciprocal_rank_fusion(keyword_hits, vector_hits, rrf_k=60)
    assert merged[0]["_id"] == "b"
    assert {item["_id"] for item in merged} == {"a", "b", "c"}
