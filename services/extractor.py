from __future__ import annotations
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader

class PDFExtractor:
    """Extract text per page. Works for text-based PDFs only."""

    def extract_pages(self, pdf_path: Path) -> List[Dict]:
        reader = PdfReader(str(pdf_path))
        pages: List[Dict] = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": i, "text": text})
        return pages
