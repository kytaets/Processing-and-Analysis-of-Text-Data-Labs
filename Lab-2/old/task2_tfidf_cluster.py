# task2_tfidf_cluster.py
# Завдання 2: Представити корпус як TF-IDF
#   + побудувати таблицю TF-IDF
#   + кластеризувати документи за допомогою ієрархічної агломеративної кластеризації
#   + вивести результат (документ → кластер)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer   # будує TF-IDF модель
from sklearn.cluster import AgglomerativeClustering            # ієрархічна кластеризація

from utils import (
    load_corpus_lines,    # читає і очищує текст з файлу
    matrix_to_table,      # перетворює матрицю в таблицю
    print_table,          # виводить таблицю в консоль
    save_tables_to_excel, # зберігає в Excel
)

DOC_PATH = "doc11.txt"
OUTPUT_EXCEL = "output/task2_tfidf_cluster_results.xlsx"

# Кількість кластерів на які розбиваємо документи
# (обираємо 3 — умовно: наприклад "морська тематика", "технологічна", "інша")
N_CLUSTERS = 3


def main():
    # ── Крок 1: завантаження корпусу ──────────────────────────────────────────
    docs = load_corpus_lines(DOC_PATH)
    print(f"Number of documents in corpus: {len(docs)}")

    # ── Крок 2: побудова TF-IDF моделі ────────────────────────────────────────
    # TF-IDF = Term Frequency × Inverse Document Frequency
    #
    # TF  (Term Frequency)        — як часто слово зустрічається в ЦЬОМУ документі
    # IDF (Inverse Doc Frequency) — наскільки слово РІДКІСНЕ в усьому корпусі
    #
    # Якщо слово часте в одному документі, але рідкісне в інших → висока вага (важливе)
    # Якщо слово є майже в кожному документі (як "the") → низька вага (неважливе)
    #
    # min_df=1        — включати слово якщо воно є хоча б в 1 документі
    # stop_words      — видаляємо стоп-слова
    tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words="english")

    # Результат: розріджена матриця з дробовими числами (0.0 — 1.0)
    # замість цілих чисел як у BoW
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)

    # Перетворюємо в зручний DataFrame
    tfidf_df = matrix_to_table(tfidf_matrix, tfidf_vectorizer.get_feature_names_out())
    print_table(tfidf_df, "TF-IDF Table")

    # ── Крок 3: ієрархічна агломеративна кластеризація ────────────────────────
    # Ідея: кожен документ — це точка в просторі (вектор TF-IDF)
    # Алгоритм знаходить "схожі" точки і об'єднує їх в групи (кластери)
    #
    # "Агломеративна" = знизу вгору:
    #   Спочатку кожен документ — окремий кластер
    #   Потім найближчі два об'єднуються в один
    #   І так далі поки не залишиться N_CLUSTERS кластерів
    #
    # metric="euclidean" — відстань між документами = звичайна евклідова відстань
    #                      (як відстань між двома точками на площині, але в 81-вимірному просторі)
    # linkage="ward"     — метод об'єднання кластерів: мінімізує загальну дисперсію всередині кластерів
    #                      (Ward — один з найякісніших методів злиття)
    model = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        metric="euclidean",
        linkage="ward",
    )

    # fit_predict: навчає модель і одразу повертає номер кластера для кожного документа
    # Наприклад: [1, 0, 1, 0, 1, 0, 2] — doc_1 у кластері 1, doc_2 у кластері 0 тощо
    # ВАЖЛИВО: номери кластерів (0, 1, 2) довільні — не мають змістовного значення
    labels = model.fit_predict(tfidf_matrix.toarray())  # .toarray() — розгортаємо sparse матрицю

    # Оформлюємо результат як таблицю: документ → номер кластера
    cluster_df = pd.DataFrame({"cluster": labels}, index=tfidf_df.index)

    print("\n=== Clustering Results (Document -> Cluster) ===")
    print(cluster_df)

    # ── Крок 4: збереження результатів ────────────────────────────────────────
    # Аркуш "TFIDF"       — повна TF-IDF таблиця
    # Аркуш "doc_cluster" — який документ потрапив у який кластер
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
