from app.services.chunker import chunk_text


def test_chunk_text_overlap():
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 3
    assert len(chunks[0]) == 500
    assert len(chunks[1]) == 500


def test_chunk_text_empty():
    assert chunk_text("", chunk_size=500, overlap=50) == []
