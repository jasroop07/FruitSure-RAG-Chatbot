from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def process_all_pdfs(pdf_directory: str):
    """Load all PDFs from the given directory."""
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    all_documents = []

    print(f"\n📂 Found {len(pdf_files)} PDF files in {pdf_directory}")
    for pdf_file in pdf_files:
        print(f"→ Processing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")
        except Exception as e:
            print(f"  ✗ Error loading {pdf_file.name}: {e}")

    print(f"✅ Total documents loaded: {len(all_documents)}")
    return all_documents
