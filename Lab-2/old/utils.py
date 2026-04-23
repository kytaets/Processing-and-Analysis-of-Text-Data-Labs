import os
import re
from typing import List, Optional

import pandas as pd
import nltk
from nltk.corpus import stopwords

# Завантажуємо список стоп-слів (the, is, a, in...) з бібліотеки nltk
# quiet=True — щоб не друкувало зайві повідомлення в консоль
nltk.download('stopwords', quiet=True)

# Перетворюємо список стоп-слів на set (множину) — пошук у set набагато швидший ніж у списку
STOP_WORDS = set(stopwords.words('english'))


def preprocess(text: str) -> str:
    """
    Очищує один рядок тексту перед подачею в модель.
    
    Кроки:
      1. Всі літери → малі  ("Mobile" → "mobile")
      2. Видалити URL-посилання  ("http://example.com" → " ")
      3. Видалити все, що не є літерою або пробілом  ("hello! 123" → "hello  ")
      4. Прибрати зайві пробіли  ("hello   world" → "hello world")
      5. Видалити стоп-слова  ("the cat sat" → "cat sat")
    
    Повертає: очищений рядок
    """
    # Крок 1: всі літери в нижній регістр
    text = text.lower()

    # Крок 2: видаляємо URL (http://... або www....)
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Крок 3: видаляємо все крім латинських літер і пробілів
    # [^a-z\s] означає "будь-який символ, що НЕ є літерою a-z або пробілом"
    text = re.sub(r"[^a-z\s]", " ", text)

    # Крок 4: замінюємо кілька пробілів підряд на один, прибираємо пробіли по краях
    text = re.sub(r"\s+", " ", text).strip()

    # Крок 5: розбиваємо рядок на окремі слова і видаляємо стоп-слова
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in STOP_WORDS]

    # Склеюємо слова назад в рядок через пробіл
    return " ".join(filtered_tokens)


def load_corpus_lines(path: str, encoding: str = "utf-8") -> List[str]:
    """
    Читає текстовий файл і повертає список очищених документів.
    Кожен рядок файлу = окремий документ.
    
    Аргументи:
      path     — шлях до файлу (наприклад "doc11.txt")
      encoding — кодування файлу (за замовчуванням utf-8)
    
    Повертає: список рядків після очищення через preprocess()
    """
    # Відкриваємо файл і читаємо всі рядки
    # errors="ignore" — якщо якийсь символ не вдається прочитати, просто пропускаємо його
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        lines = [line.strip() for line in f.readlines()]  # .strip() прибирає \n в кінці

    # Викидаємо порожні рядки (пусті рядки між абзацами тощо)
    docs_raw = [x for x in lines if x]

    # Застосовуємо preprocess() до кожного рядка і повертаємо результат
    return [preprocess(d) for d in docs_raw]


def matrix_to_table(matrix, feature_names, index_prefix: str = "doc_") -> pd.DataFrame:
    """
    Перетворює матрицю (результат векторизації) на зручну таблицю pandas DataFrame.
    
    Аргументи:
      matrix        — розріджена матриця від CountVectorizer або TfidfVectorizer
      feature_names — список назв стовпців (слова словника)
      index_prefix  — префікс для назв рядків (за замовчуванням "doc_")
    
    Повертає: DataFrame з рядками doc_1, doc_2, ... і стовпцями = слова
    
    Приклад результату:
          athens  mariner  mobile
    doc_1    0       2       0
    doc_2    1       0       1
    """
    # Якщо матриця розріджена (sparse) — конвертуємо в звичайний numpy масив
    arr = matrix.toarray() if hasattr(matrix, "toarray") else matrix

    # Створюємо DataFrame: стовпці = слова, рядки = документи
    df = pd.DataFrame(arr, columns=feature_names)

    # Перейменовуємо індекси: 0,1,2... → doc_1, doc_2, doc_3...
    df.index = [f"{index_prefix}{i+1}" for i in range(df.shape[0])]

    return df


def ensure_parent_dir(filepath: str) -> None:
    """
    Створює папку для файлу, якщо вона ще не існує.
    
    Наприклад: якщо filepath = "output/results.xlsx",
    то створить папку "output/" якщо її немає.
    
    Потрібно щоб уникнути помилки "папка не знайдена" при збереженні файлу.
    """
    parent = os.path.dirname(filepath)  # беремо лише шлях до папки, без імені файлу
    if parent:
        os.makedirs(parent, exist_ok=True)  # exist_ok=True — не падати якщо папка вже є


def save_tables_to_excel(
    filepath: str,
    tables: List[tuple[str, pd.DataFrame]],
) -> None:
    """
    Зберігає кілька таблиць (DataFrame) в один Excel-файл, кожна на окремому аркуші.
    
    Аргументи:
      filepath — шлях до вихідного .xlsx файлу
      tables   — список пар (назва_аркуша, таблиця)
    
    Приклад використання:
      save_tables_to_excel("output/results.xlsx", [
          ("BoW",     bow_df),
          ("Clusters", cluster_df),
      ])
    """
    # Спочатку переконуємось що папка існує
    ensure_parent_dir(filepath)

    # Відкриваємо Excel-файл для запису (engine="openpyxl" — бібліотека для роботи з .xlsx)
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for sheet_name, df in tables:
            # Записуємо кожну таблицю на окремий аркуш, зберігаючи індекс (назви рядків)
            df.to_excel(writer, sheet_name=sheet_name, index=True)


def print_table(
    df: pd.DataFrame,
    title: str,
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None
) -> None:
    """
    Красиво виводить таблицю в консоль з заголовком.
    Якщо таблиця велика — можна обмежити кількість рядків/стовпців для виводу.
    
    Аргументи:
      df       — таблиця для виводу
      title    — заголовок (виводиться між === ===)
      max_rows — скільки рядків показати максимум (None = всі)
      max_cols — скільки стовпців показати максимум (None = всі)
    """
    print(f"\n=== {title} ===")

    view = df

    # Обрізаємо рядки якщо задано ліміт
    if max_rows is not None:
        view = view.head(max_rows)

    # Обрізаємо стовпці якщо задано ліміт
    if max_cols is not None:
        view = view.iloc[:, :max_cols]

    print(view)

    # Якщо таблиця була обрізана — повідомляємо про реальний розмір
    if (max_rows is not None and df.shape[0] > max_rows) or \
       (max_cols is not None and df.shape[1] > max_cols):
        print(f"... повний розмір таблиці: {df.shape[0]} рядків x {df.shape[1]} стовпців")
