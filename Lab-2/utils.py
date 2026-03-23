import os
import re
from typing import List, Optional

import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

def preprocess(text: str) -> str:
    text = text.lower()                         
    text = re.sub(r"http\S+|www\.\S+", " ", text)  
    text = re.sub(r"[^a-z\s]", " ", text)       
    text = re.sub(r"\s+", " ", text).strip()    
    
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in STOP_WORDS]
    
    return " ".join(filtered_tokens)

def load_corpus_lines(path: str, encoding: str = "utf-8") -> List[str]:
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        lines = [line.strip() for line in f.readlines()]

    docs_raw = [x for x in lines if x]          
    return [preprocess(d) for d in docs_raw]   

def matrix_to_table(matrix, feature_names, index_prefix: str = "doc_") -> pd.DataFrame:
    arr = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    df = pd.DataFrame(arr, columns=feature_names)
    df.index = [f"{index_prefix}{i+1}" for i in range(df.shape[0])]
    return df

def ensure_parent_dir(filepath: str) -> None:
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

def save_tables_to_excel(
    filepath: str,
    tables: List[tuple[str, pd.DataFrame]],
) -> None:
    ensure_parent_dir(filepath)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for sheet_name, df in tables:
            df.to_excel(writer, sheet_name=sheet_name, index=True)

def print_table(
    df: pd.DataFrame,
    title: str,
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None
) -> None:
    print(f"\n=== {title} ===")

    view = df
    if max_rows is not None:
        view = view.head(max_rows)
    if max_cols is not None:
        view = view.iloc[:, :max_cols]

    print(view)

    if (max_rows is not None and df.shape[0] > max_rows) or (max_cols is not None and df.shape[1] > max_cols):
        print(f"... повний розмір таблиці: {df.shape[0]} рядків x {df.shape[1]} стовпців")