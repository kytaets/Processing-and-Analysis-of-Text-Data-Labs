import pandas as pd
import numpy as np
import random
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel  # LDA модель і метрика оцінки якості
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


def preprocess(text, stop_words, punctuation):
    """
    Очищує один текст і повертає список корисних токенів (слів).
    
    Кроки:
      1. str(text).lower() — переводимо в рядок і всі літери в нижній регістр
      2. word_tokenize()   — розбиваємо текст на окремі слова і розділові знаки
                             "Hello, world!" → ["Hello", ",", "world", "!"]
      3. Фільтруємо — залишаємо тільки слова які:
           - не є стоп-словами (the, is, a...)
           - не є розділовими знаками (.,!?...)
           - довші за 2 символи (прибираємо "it", "of", "in" тощо)
    
    Повертає: список очищених слів, наприклад ["laptop", "broken", "screen"]
    """
    tokens = word_tokenize(str(text).lower())
    return [word for word in tokens
            if word not in stop_words
            and word not in punctuation
            and len(word) > 2]


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    """
    Перебирає різну кількість тем (від start до limit з кроком step),
    навчає LDA модель для кожної і обчислює метрику узгодженості (coherence).
    
    Coherence (C_v) — оцінює наскільки слова всередині теми "пасують" одне до одного.
    Чим вище значення → тим більш зв'язні і зрозумілі теми.
    
    Аргументи:
      dictionary — словник (унікальні слова корпусу)
      corpus     — корпус у форматі BoW (список векторів документів)
      texts      — оригінальні токенізовані документи (для CoherenceModel)
      limit      — максимальна кількість тем (не включно)
      start      — мінімальна кількість тем (за замовчуванням 2)
      step       — крок перебору (за замовчуванням 2)
    
    Повертає: (список моделей, список значень coherence)
    """
    coherence_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        # Навчаємо LDA модель з поточною кількістю тем
        # passes=10      — кількість проходів по всьому корпусу (більше = якісніше)
        # random_state=0 — фіксуємо випадковість для відтворюваності
        model = LdaModel(
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=10,
            random_state=0
        )
        model_list.append(model)

        # Обчислюємо C_v coherence — популярна метрика якості тем
        # Значення від 0 до 1: ~0.4 прийнятно, ~0.6+ добре
        coherencemodel = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_values.append(coherencemodel.get_coherence())
        print(f"Кількість тем: {num_topics}, Оцінка узгодженості (C_v): {coherence_values[-1]:.4f}")

    return model_list, coherence_values


if __name__ == '__main__':
    nltk.download('punkt', quiet=True)      # токенізатор
    nltk.download('stopwords', quiet=True)  # список стоп-слів

    # ── Крок 1: Зчитування і семплінг даних ──────────────────────────────────
    print("Зчитування даних...")
    file_path = 'ecommerceDataset4.csv'
    df = pd.read_csv(file_path)

    text_column = 'text'      # колонка з текстом товару/відгуку
    class_column = 'category' # колонка з категорією (для стратифікованого семплінгу)

    # Беремо 20% документів з КОЖНОЇ категорії рівномірно
    # groupby + apply(sample) — щоб не взяти випадково всі дані з однієї категорії
    # include_groups=False    — не включати колонку групування в результат
    df_sampled = df.groupby(class_column, group_keys=False).apply(
        lambda x: x.sample(frac=0.2, random_state=42),
        include_groups=False
    )

    # Беремо тільки тексти, видаляємо пусті значення
    documents = df_sampled[text_column].dropna().tolist()
    print(f"Відібрано документів для аналізу: {len(documents)}")

    # ── Крок 2: Попередня обробка ─────────────────────────────────────────────
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)  # { '.', ',', '!', '?', ... }

    print("Попередня обробка документів...")
    # Фільтруємо документи коротші за 10 символів (занадто короткі — марні)
    processed_docs = [
        preprocess(doc, stop_words, punctuation)
        for doc in documents
        if len(str(doc)) > 10
    ]

    # ── Крок 3: Побудова словника і BoW корпусу ───────────────────────────────
    # Dictionary — зіставляє кожне унікальне слово з числовим ID
    # Наприклад: {"laptop": 0, "screen": 1, "broken": 2, ...}
    dictionary = corpora.Dictionary(processed_docs)

    # Фільтруємо рідкісні та надто часті слова:
    # no_below=5   — видаляємо слова що зустрічаються менш ніж у 5 документах
    # no_above=0.5 — видаляємо слова що є більш ніж у 50% документів
    # (рідкісні — шум; надто часті — нічого не означають)
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    # doc2bow — перетворює список слів на список пар (id_слова, кількість)
    # Наприклад: ["laptop", "broken"] → [(0, 1), (2, 1)]
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # ── Крок 4: Пошук оптимальної кількості тем ──────────────────────────────
    print("\nПошук оптимальної кількості тем...")
    # Перебираємо: 2, 4, 6, 8 тем і обираємо з найвищим coherence
    model_list, coherence_values = compute_coherence_values(
        dictionary=dictionary,
        corpus=bow_corpus,
        texts=processed_docs,
        start=2,
        limit=10,
        step=2
    )

    # argmax — індекс максимального значення у списку coherence_values
    # Це і є індекс найкращої моделі
    optimal_index = np.argmax(coherence_values)
    best_lda_model = model_list[optimal_index]
    optimal_num_topics = best_lda_model.num_topics
    print(f"\nОптимальна кількість тем: {optimal_num_topics}")

    # ── Крок 5: Виведення тем ─────────────────────────────────────────────────
    # LDA визначає теми як розподіл ймовірностей по словах
    # print_topics виводить 5 найважливіших слів для кожної теми з вагами
    # Наприклад: Topic 0: 0.05*"phone" + 0.04*"battery" + 0.03*"screen" ...
    print("\nТерми для кожної теми:")
    topics = best_lda_model.print_topics(num_words=5)
    for topic in topics:
        print(topic)

    # ── Крок 6: Аналіз випадкових документів ─────────────────────────────────
    # Для кожного документа LDA дає розподіл ймовірностей по темах
    # Наприклад: [(0, 0.72), (2, 0.18), (1, 0.10)] — документ найбільш схожий на тему 0
    print("\nАналіз трьох випадкових документів:")
    random_indices = random.sample(range(len(bow_corpus)), 3)

    for idx in random_indices:
        # get_document_topics — повертає список (номер_теми, ймовірність) для документа
        doc_topics = best_lda_model.get_document_topics(bow_corpus[idx])

        # Сортуємо від найвищої ймовірності до найнижчої
        doc_topics.sort(key=lambda x: x[1], reverse=True)

        # Беремо тему з найвищою ймовірністю
        most_important_topic = doc_topics[0][0]
        topic_probability = doc_topics[0][1]

        # Виводимо перші 10 слів документа і його найважливішу тему
        print(f"\nДокумент #{idx} (перші 50 символів): '{' '.join(processed_docs[idx][:10])}...'")
        print(f"Найважливіша тема: {most_important_topic} (Ймовірність: {topic_probability:.4f})")
