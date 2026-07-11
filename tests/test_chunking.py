import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.chunking import chunk_pages
from ingestion.loaders import PageText


def test_short_page_produces_single_chunk():
    pages = [PageText(doc_id="doc1", page_number=1, text="This is a short page.")]
    chunks = chunk_pages(pages, chunk_size=800, overlap=120)
    assert len(chunks) == 1
    assert chunks[0].doc_id == "doc1"
    assert chunks[0].page_number == 1


def test_long_page_splits_into_multiple_chunks():
    paragraph = "Sentence about revenue and risk factors. " * 50  # ~2150 chars
    pages = [PageText(doc_id="doc1", page_number=1, text=paragraph)]
    chunks = chunk_pages(pages, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 500 + 50 + 5  # allow small slack from overlap stitching


def test_overlap_creates_shared_text_between_adjacent_chunks():
    paragraph = "A. " * 400
    pages = [PageText(doc_id="doc1", page_number=1, text=paragraph)]
    chunks = chunk_pages(pages, chunk_size=300, overlap=60)
    assert len(chunks) >= 2
    # the tail of chunk N should reappear at the head of chunk N+1
    tail = chunks[0].text[-30:]
    assert tail[:10] in chunks[1].text


def test_empty_page_produces_no_chunks():
    pages = [PageText(doc_id="doc1", page_number=1, text="   ")]
    chunks = chunk_pages(pages)
    assert chunks == []


def test_metadata_populated():
    pages = [PageText(doc_id="10k_2024", page_number=7, text="Risk factors include market volatility.")]
    chunks = chunk_pages(pages)
    assert chunks[0].metadata["doc_id"] == "10k_2024"
    assert chunks[0].metadata["page_number"] == 7
