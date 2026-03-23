# task3_word2vec.py
# 3) Word2Vec:
#    - вивести вектори слів "mobile", "athens"
#    - знайти подібні слова до "mobile", "athens"

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from utils import load_corpus_lines, ensure_parent_dir, save_tables_to_excel

DOC_PATH = "doc11.txt"
OUTPUT_EXCEL = "output/task3_word2vec_results.xlsx"

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 1
SG = 1          # 1 = Skip-gram, 0 = CBOW
EPOCHS = 50
TOPN = 10

def vector_to_df(word: str, vec: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([vec], index=[word])

def similar_to_df(word: str, sim_list) -> pd.DataFrame:
    return pd.DataFrame(sim_list, columns=["similar_word", "similarity"]).set_index("similar_word")

def main():
    docs = load_corpus_lines(DOC_PATH)
    print(f"Number of documents in corpus: {len(docs)}")

    tokenized_docs = [doc.split() for doc in docs if doc.strip()]

    model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=4,
        sg=SG,
        epochs=EPOCHS,
        seed=42
    )

    target_words = ["mobile", "athens"]
    tables_to_save = []

    for w in target_words:
        if w not in model.wv:
            print(f"\nThe word '{w}' is not present in the Word2Vec vocabulary.")
            note_df = pd.DataFrame({"note": [f"'{w}' not found in vocabulary"]}, index=[w])
            tables_to_save.append((f"{w}_note", note_df))
            continue

        vec = model.wv[w]

        print(f"\nVector for word '{w}' (first 10 components):")
        print(np.round(vec[:10], 6))

        sim = model.wv.most_similar(w, topn=TOPN)
        print(f"\nTop {TOPN} similar words to '{w}':")
        for sw, score in sim:
            print(f"  {sw:20s} {score:.4f}")

        vec_df = vector_to_df(w, vec)
        sim_df = similar_to_df(w, sim)

        tables_to_save.append((f"{w}_vector", vec_df))
        tables_to_save.append((f"{w}_similar", sim_df))

    ensure_parent_dir(OUTPUT_EXCEL)
    save_tables_to_excel(OUTPUT_EXCEL, tables=tables_to_save)
    print(f"\nResults saved to '{OUTPUT_EXCEL}'")

if __name__ == "__main__":
    main()
