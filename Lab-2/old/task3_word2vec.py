# task3_word2vec.py
# Завдання 3: Модель Word2Vec
#   - вивести вектори слів "mobile", "athens"
#   - знайти найбільш подібні слова до "mobile", "athens"

import numpy as np
import pandas as pd
from gensim.models import Word2Vec  # бібліотека для навчання Word2Vec

from utils import load_corpus_lines, ensure_parent_dir, save_tables_to_excel

DOC_PATH = "doc11.txt"
OUTPUT_EXCEL = "output/task3_word2vec_results.xlsx"

# ── Гіперпараметри моделі Word2Vec ────────────────────────────────────────────
VECTOR_SIZE = 100   # розмірність вектора кожного слова (100 чисел на слово)
WINDOW = 5          # розмір "вікна контексту": скільки сусідніх слів враховуємо
                    # Приклад при window=2: "The [quick brown fox] jumps" — аналізуємо 2 слова зліва і справа
MIN_COUNT = 1       # мінімальна кількість входжень слова щоб воно потрапило в словник
                    # MIN_COUNT=1 — включаємо навіть слова що зустрічаються 1 раз
SG = 1              # архітектура моделі:
                    #   SG=1 → Skip-gram: дано слово → передбачаємо його сусідів (краще для рідкісних слів)
                    #   SG=0 → CBOW: дано сусіди → передбачаємо слово (швидше, краще для частих слів)
EPOCHS = 50         # кількість проходів по всьому корпусу під час навчання
                    # Більше епох = краще навчання, але довше
TOPN = 10           # скільки найбільш схожих слів виводити


def vector_to_df(word: str, vec: np.ndarray) -> pd.DataFrame:
    """
    Перетворює вектор слова (масив чисел) на DataFrame для збереження в Excel.
    
    Аргументи:
      word — слово (буде індексом рядка)
      vec  — numpy масив з 100 числами (вектор слова)
    
    Повертає: DataFrame з одним рядком (індекс = слово) і 100 стовпцями
    """
    # [vec] — обертаємо вектор у список щоб DataFrame мав один рядок
    return pd.DataFrame([vec], index=[word])


def similar_to_df(word: str, sim_list) -> pd.DataFrame:
    """
    Перетворює список схожих слів на DataFrame для збереження в Excel.
    
    Аргументи:
      word     — вихідне слово (для контексту)
      sim_list — список пар [(слово, схожість), ...] від model.wv.most_similar()
                 Наприклад: [("people", 0.797), ("in", 0.786), ...]
    
    Повертає: DataFrame з колонками similar_word та similarity
    """
    return (
        pd.DataFrame(sim_list, columns=["similar_word", "similarity"])
        .set_index("similar_word")  # робимо слово індексом для зручності
    )


def main():
    # ── Крок 1: завантаження і токенізація корпусу ────────────────────────────
    docs = load_corpus_lines(DOC_PATH)
    print(f"Number of documents in corpus: {len(docs)}")

    # Word2Vec потребує список списків слів (а не список рядків)
    # doc.split() — розбиває рядок "hello world" на ["hello", "world"]
    # Результат: [["word1", "word2", ...], ["word3", ...], ...]
    tokenized_docs = [doc.split() for doc in docs if doc.strip()]

    # ── Крок 2: навчання моделі Word2Vec ──────────────────────────────────────
    # Word2Vec — нейронна мережа, яка навчається передбачати контекст слів
    # Після навчання кожне слово має унікальний вектор з 100 чисел
    # Слова зі схожим контекстом (сусідами) → схожі вектори
    #
    # seed=42 — фіксуємо випадковість для відтворюваності результатів
    #           (щоб кожен запуск давав однаковий результат)
    model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=4,       # кількість паралельних потоків навчання
        sg=SG,
        epochs=EPOCHS,
        seed=42
    )

    # ── Крок 3: аналіз цільових слів ─────────────────────────────────────────
    target_words = ["mobile", "athens"]
    tables_to_save = []

    for w in target_words:
        # Перевіряємо чи слово взагалі є в словнику моделі
        if w not in model.wv:
            print(f"\nThe word '{w}' is not present in the Word2Vec vocabulary.")
            note_df = pd.DataFrame({"note": [f"'{w}' not found in vocabulary"]}, index=[w])
            tables_to_save.append((f"{w}_note", note_df))
            continue  # переходимо до наступного слова

        # Отримуємо вектор слова — масив з 100 чисел
        # model.wv — "word vectors", сховище всіх навчених векторів
        vec = model.wv[w]

        # Виводимо лише перші 10 компонентів (бо всі 100 — забагато для консолі)
        print(f"\nVector for word '{w}' (first 10 components):")
        print(np.round(vec[:10], 6))  # round до 6 знаків після коми для читабельності

        # Знаходимо TOPN=10 найбільш схожих слів
        # Схожість вимірюється косинусною відстанню між векторами (від -1 до 1)
        # Чим ближче до 1 — тим більш схожі слова
        sim = model.wv.most_similar(w, topn=TOPN)

        print(f"\nTop {TOPN} similar words to '{w}':")
        for sw, score in sim:
            print(f"  {sw:20s} {score:.4f}")

        # Оформлюємо результати для збереження в Excel
        vec_df = vector_to_df(w, vec)    # таблиця з вектором (100 чисел)
        sim_df = similar_to_df(w, sim)   # таблиця зі схожими словами

        # Додаємо обидві таблиці до списку для збереження
        tables_to_save.append((f"{w}_vector", vec_df))   # аркуш "mobile_vector" або "athens_vector"
        tables_to_save.append((f"{w}_similar", sim_df))  # аркуш "mobile_similar" або "athens_similar"

    # ── Крок 4: збереження результатів ────────────────────────────────────────
    # Excel-файл матиме 4 аркуші:
    #   mobile_vector  — вектор слова "mobile" (100 чисел)
    #   mobile_similar — 10 найсхожіших слів до "mobile"
    #   athens_vector  — вектор слова "athens"
    #   athens_similar — 10 найсхожіших слів до "athens"
    ensure_parent_dir(OUTPUT_EXCEL)
    save_tables_to_excel(OUTPUT_EXCEL, tables=tables_to_save)
    print(f"\nResults saved to '{OUTPUT_EXCEL}'")


if __name__ == "__main__":
    main()
