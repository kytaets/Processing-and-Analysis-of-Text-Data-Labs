# task1_bow.py
# Завдання 1: Представити корпус як модель "Сумка слів" (Bag of Words)
#   - вивести модель у вигляді таблиці
#   - вивести вектор для слова "mariner"

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer  # будує модель Bag of Words

from utils import (
    load_corpus_lines,    # читає і очищує текст з файлу
    matrix_to_table,      # перетворює матрицю в красиву таблицю
    print_table,          # виводить таблицю в консоль
    save_tables_to_excel, # зберігає результати в Excel
)

# Шлях до вхідного файлу з текстом
DOC_PATH = "doc11.txt"

# Шлях куди зберегти результати
OUTPUT_EXCEL = "output/task1_bow_results.xlsx"


def main():
    # ── Крок 1: завантаження корпусу ──────────────────────────────────────────
    # Читаємо файл — кожен рядок стає окремим документом
    # Функція також автоматично очищує текст (lowercase, видалення стоп-слів тощо)
    docs = load_corpus_lines(DOC_PATH)
    print(f"Number of documents in corpus: {len(docs)}")

    # ── Крок 2: побудова моделі Bag of Words ──────────────────────────────────
    # CountVectorizer — рахує скільки разів кожне слово зустрічається в документі
    # min_df=1       — включати слово якщо воно є хоча б в 1 документі (тобто всі слова)
    # stop_words     — додатково видаляємо стоп-слова вже на рівні векторизатора
    vectorizer = CountVectorizer(min_df=1, stop_words="english")

    # fit_transform:
    #   fit      — вивчає словник (всі унікальні слова в корпусі)
    #   transform — перетворює кожен документ на вектор чисел
    # Результат: розріджена матриця розміром (кількість_документів × розмір_словника)
    bow_matrix = vectorizer.fit_transform(docs)

    # Перетворюємо матрицю на зручний DataFrame
    # get_feature_names_out() повертає список всіх слів словника (назви стовпців)
    bow_df = matrix_to_table(bow_matrix, vectorizer.get_feature_names_out())

    # Виводимо таблицю в консоль
    print_table(bow_df, "Bag of Words Table (BoW)")

    # ── Крок 3: вектор для слова "mariner" ────────────────────────────────────
    word = "mariner"

    if word in bow_df.columns:
        # Беремо стовпець "mariner" — це і є вектор слова
        # Кожне значення показує скільки разів "mariner" є в кожному документі
        # Наприклад: [2, 0, 1, 0, 2, 0, 0] — у doc_1 зустрічається 2 рази, у doc_2 — 0, тощо
        mariner_vec = bow_df[word].to_numpy()

        # Оформлюємо вектор як окрему таблицю для збереження в Excel
        mariner_df = pd.DataFrame({"mariner_count": mariner_vec}, index=bow_df.index)

        print(f"\nVector for word '{word}' (across documents):")
        print(mariner_vec)
    else:
        # Якщо слово взагалі не знайдено в словнику — повідомляємо про це
        mariner_df = pd.DataFrame(
            {"mariner_count": ["WORD NOT FOUND"] * len(bow_df)},
            index=bow_df.index
        )
        print(f"\nThe word '{word}' is NOT present in the BoW vocabulary.")

    # ── Крок 4: збереження результатів ────────────────────────────────────────
    # Зберігаємо дві таблиці в один Excel-файл на різних аркушах:
    #   аркуш "BoW"            — повна таблиця Bag of Words
    #   аркуш "mariner_vector" — вектор слова mariner
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
