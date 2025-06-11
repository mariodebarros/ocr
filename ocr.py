import argparse
#<<<<<<< 624zzy-codex/ajustar-código-para-ler-imagem-ou-pdf-e-extrair-campos
import json
import re
import torch
import os
import json
import re
import torch
import pandas as pd
#>>>>>>> master

import tkinter as tk
from tkinter import filedialog
from PIL import Image
from pdf2image import convert_from_path
#from PyPDF2 import PdfReader
from pathlib import Path
from typing import List

from transformers import DonutProcessor, VisionEncoderDecoderModel

# ABRE O ARQUIVO A SER LIDO
def read_file_path_via_dialog():
    # Step 1: Create a hidden root window (we don’t need to show a full GUI)
    root = tk.Tk()
    root.withdraw()  # ocultar janela principal
    root.after(0, lambda: root.attributes('-topmost', False))

    # Step 2: Open a file‐selection dialog and let the user pick a file
    file_path = filedialog.askopenfilename(
        title="Selecione um arquivo",
        filetypes=[("All files", "*.*")]  # opcional: especifique tipos, ex: [("Text files", "*.txt")]
    )

    # If the user canceled, file_path virá como string vazia
    if not file_path:
        print("Nenhum arquivo selecionado.")
        return
    
    # Comentado: leitura de arquivo para debug

    return file_path

def parse_nfe(file_path: str) -> dict:
    """Return the parsed JSON representation of an NF-e or cupom fiscal."""
    path = Path(file_path).expanduser()
    pages = _load_pages(path)

    results = []
    for img in pages:
        results.append(_parse_page(img))

    if len(results) == 1:
        return results[0]
    return {"pages": results}

def _load_pages(path: Path) -> List[Image.Image]:
    """Return a list of images (pdf pages or single image)."""
    if path.suffix.lower() == ".pdf":
        return _pdf_to_images(path)
    return [Image.open(path)]

def _pdf_to_images(pdf_path: Path, dpi: int = 200) -> List[Image.Image]:
    """Convert each page of a PDF to PIL.Image."""
    return convert_from_path(str(pdf_path), dpi=dpi)

def _parse_page(img: Image.Image) -> dict:
    """Run Donut on a single page and decode JSON."""
    inputs  = processor(img, TASK_PROMPT, return_tensors="pt").to(device)
    output  = model.generate(**inputs, max_length=512)
    result  = processor.decode(output[0], skip_special_tokens=True)
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        # remove trailing commas, etc. if model produced "almost-JSON"
        cleaned = re.sub(r",(\s*[}\]])", r"\1", result)
        return json.loads(cleaned)
    
def _items_from_json(page_json: dict) -> List[dict]:
    """
    Best-effort extraction of the Items array regardless of field name.

    The preset checkpoint stores items under ...["Items"] but custom models
    may choose e.g. ["Invoice"]["Items"]. Adjust here if you train your own.
    """
    # search recursively for a key that looks like 'items'
    stack = [page_json]
    while stack:
        obj = stack.pop()
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() == "items" and isinstance(v, list):
                    return v
                stack.append(v)
        elif isinstance(obj, list):
            stack.extend(obj)
    return []

# ---- model & processor ----------------------------------------------------


#<<<<<<< 624zzy-codex/ajustar-código-para-ler-imagem-ou-pdf-e-extrair-campos
CHECKPOINT = "naver-clova-ix/donut-base-finetuned-cord-v2"  # public invoice model
TASK_PROMPT = "<s_cord-v2>"
PROC_CKPT = CHECKPOINT

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = DonutProcessor.from_pretrained(CHECKPOINT, use_fast=False)
#>>>>>>> master
model = VisionEncoderDecoderModel.from_pretrained(CHECKPOINT).to(device)
model.eval()
"""if TASK_PROMPT not in processor.tokenizer.get_vocab():
    processor.tokenizer.add_tokens([TASK_PROMPT])
    model.resize_token_embeddings(len(processor.tokenizer))"""



def main():
    parser = argparse.ArgumentParser(description="Extrai campos de notas ou cupons fiscais")
    parser.add_argument("file", nargs="?", help="Caminho para imagem ou PDF")
    args = parser.parse_args()

#<<<<<<< 624zzy-codex/ajustar-código-para-ler-imagem-ou-pdf-e-extrair-campos
    file_path = args.file if args.file else read_file_path_via_dialog()
    if not file_path:
        return

    print(f"Arquivo selecionado: {file_path}")
    data = parse_nfe(file_path)
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()