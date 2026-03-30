"""Load PDF documents"""
from pypdf import PdfReader

def load_pdf(filepath: str) -> str:
    """Extract text from PDF"""
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text