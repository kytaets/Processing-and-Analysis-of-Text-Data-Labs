# task1_bow.py
# 1) Корпус як "Сумка слів" (Bag of Words)
#    - вивести модель у вигляді таблиці
#    - вивести вектор для слова "mariner"

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from utils import (
    load_corpus_lines,
    matrix_to_table,
    print_table,
    save_tables_to_excel,
)

DOC_PATH = "doc11.txt"
OUTPUT_EXCEL = "output/task1_bow_results.xlsx"

def main():
    docs = load_corpus_lines(DOC_PATH)
    print(f"Number of documents in corpus: {len(docs)}")

    vectorizer = CountVectorizer(min_df=1, stop_words="english")
    bow_matrix = vectorizer.fit_transform(docs)

    bow_df = matrix_to_table(bow_matrix, vectorizer.get_feature_names_out())

    print_table(bow_df, "Bag of Words Table (BoW)")

    word = "mariner"
    if word in bow_df.columns:
        mariner_vec = bow_df[word].to_numpy()
        mariner_df = pd.DataFrame({"mariner_count": mariner_vec}, index=bow_df.index)

        print(f"\nVector for word '{word}' (across documents):")
        print(mariner_vec)
    else:
        mariner_df = pd.DataFrame({"mariner_count": ["WORD NOT FOUND"] * len(bow_df)}, index=bow_df.index)
        print(f"\nThe word '{word}' is NOT present in the BoW vocabulary.")

    save_tables_to_excel(
        OUTPUT_EXCEL,
        tables=[
            ("BoW", bow_df),
            ("mariner_vector", mariner_df),
        ],
    )
    print(f"\nResults saved to '{OUTPUT_EXCEL}'")

if __name__ == "__main__":
    main()