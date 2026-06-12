from kairorag.cloud.manifest import (
    CloudChunkRecord,
    CloudIndexManifest,
    find_chunk,
    load_manifest,
    save_manifest,
)


def test_cloud_manifest_save_load_and_find_chunk(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = CloudIndexManifest(
        created_at="2026-06-12T00:00:00+00:00",
        embedding_model="text-embedding-3-small",
        vector_store_provider="qdrant",
        vector_collection="kairo_chunks",
        keyword_search_provider="bm25",
        chunk_count=1,
        chunks=[
            CloudChunkRecord(
                chunk_id="doc-1_chunk_000",
                doc_id="doc-1",
                title="测试文档",
                text="完整 chunk 文本",
                source_type="note",
                metadata={"company": "Kairo"},
            )
        ],
    )

    save_manifest(manifest, path)
    loaded = load_manifest(path)

    assert loaded.chunk_count == 1
    assert loaded.chunks[0].text == "完整 chunk 文本"
    assert find_chunk(loaded, "doc-1_chunk_000") == loaded.chunks[0]
    assert find_chunk(loaded, "missing") is None
