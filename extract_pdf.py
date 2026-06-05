from pathlib import Path
from PyPDF2 import PdfReader
path = Path('mini projet nlp (1).pdf')
reader = PdfReader(path)
for i, page in enumerate(reader.pages):
    print(f'--- PAGE {i+1} ---')
    print(page.extract_text())
