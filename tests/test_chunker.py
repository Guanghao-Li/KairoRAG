from kairorag.ingestion.chunker import chunk_documents
from kairorag.schemas import DocumentRecord


def test_chunker_stable_ids_and_overlap():
    doc = DocumentRecord(
        doc_id="doc_1",
        source_type="company_doc",
        title="Doc",
        text=" ".join(f"token{i}" for i in range(20)),
        metadata={"verification_status": "active"},
    )
    chunks = chunk_documents([doc], chunk_size=8, overlap=2)
    assert [chunk.chunk_id for chunk in chunks][:2] == ["doc_1_chunk_000", "doc_1_chunk_001"]
    assert chunks[0].metadata["title"] == "Doc"
    assert chunks[0].token_count > 0
    assert "token6" in chunks[1].text

