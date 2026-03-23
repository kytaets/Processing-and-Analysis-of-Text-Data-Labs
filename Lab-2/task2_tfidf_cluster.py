# task2_tfidf_cluster.py
# 2) Корпус як TF-IDF (таблиця)
#    + ієрархічна агломеративна кластеризація
#    + вивести (документ-кластер)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

from utils import (
    load_corpus_lines,
    matrix_to_table,
    print_table,
    save_tables_to_excel,
)

DOC_PATH = "doc11.txt"
OUTPUT_EXCEL = "output/task2_tfidf_cluster_results.xlsx"
N_CLUSTERS = 3

def main():
    docs = load_corpus_lines(DOC_PATH)
    print(f"Number of documents in corpus: {len(docs)}")

    tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)

    tfidf_df = matrix_to_table(tfidf_matrix, tfidf_vectorizer.get_feature_names_out())

    print_table(tfidf_df, "TF-IDF Table")

    model = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        metric="euclidean",
        linkage="ward",
    )
    labels = model.fit_predict(tfidf_matrix.toarray())

    cluster_df = pd.DataFrame({"cluster": labels}, index=tfidf_df.index)

    print("\n=== Clustering Results (Document -> Cluster) ===")
    print(cluster_df)

    save_tables_to_excel(
        OUTPUT_EXCEL,
        tables=[
            ("TFIDF", tfidf_df),
            ("doc_cluster", cluster_df),
        ],
    )
    print(f"\nResults saved to '{OUTPUT_EXCEL}'")

if __name__ == "__main__":
    main()