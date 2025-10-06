import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
import shutil

INDEX_DIR = "bm25_index"

def prepare_bm25_index(pdf_path: str):
    # Clear any existing index
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    os.mkdir(INDEX_DIR)

    schema = Schema(content=TEXT(stored=True), path=ID(stored=True))
    ix = create_in(INDEX_DIR, schema)

    # Decide if it's a URL or a local file
    if pdf_path.startswith("http://") or pdf_path.startswith("https://"):
        temp_path = "temp_downloaded.pdf"
        response = requests.get(pdf_path)
        with open(temp_path, "wb") as f:
            f.write(response.content)
    else:
        # Local file case (e.g. /tmp/tmpxxxxx.pdf on Render)
        temp_path = pdf_path

    # Load PDF
    loader = PyPDFLoader(temp_path)
    documents = loader.load()
    full_text = "\n\n".join([doc.page_content for doc in documents])

    # Add to BM25 index
    writer = ix.writer()
    writer.add_document(content=full_text, path="doc1")
    writer.commit()

    # Clean up only if we downloaded from a URL
    if pdf_path.startswith("http://") or pdf_path.startswith("https://"):
        os.remove(temp_path)

def bm25_search(query, k=3, index_dir=INDEX_DIR):
    ix = open_dir(index_dir)
    with ix.searcher() as searcher:
        parser = QueryParser("content", ix.schema)
        parsed_query = parser.parse(query)
        results = searcher.search(parsed_query, limit=k)
        return [r['content'] for r in results]
