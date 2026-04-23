import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from gensim.models import FastText
from nltk.tokenize import word_tokenize
import nltk

# Завантажуємо токенізатор punkt — він вміє розбивати текст на слова
# ("Hello world!" → ["Hello", "world", "!"])
nltk.download('punkt', quiet=True)


# ── Крок 1: Завантаження і підготовка даних ───────────────────────────────────

# Читаємо CSV-файл з IT-тікетами (заявками в службу підтримки)
# Колонки: 'Document' — текст тікета, 'Topic_group' — категорія/тема
df = pd.read_csv("IT_tickets2.csv")

# Видаляємо рядки де відсутній текст або мітка категорії
# (не можна навчати модель на неповних даних)
df = df.dropna(subset=['Document', 'Topic_group'])

# Вхідні дані X — тексти тікетів
# Цільова змінна y — категорія кожного тікета (те, що треба передбачити)
texts = df['Document'].astype(str)
labels = df['Topic_group']

# Ділимо дані на тренувальну (70%) і тестову (30%) вибірки
# test_size=0.3  — 30% даних залишаємо для перевірки
# random_state=42 — фіксуємо випадковість (щоб результат був однаковим при кожному запуску)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)


# ── Крок 2: Модель BoW (Bag of Words) — базові класифікатори ─────────────────

# CountVectorizer перетворює текст на вектор чисел (Bag of Words)
# fit_transform на тренувальних — вивчає словник І перетворює
# transform на тестових     — тільки перетворює (словник вже відомий!)
# ВАЖЛИВО: не можна робити fit на тестових даних — це "витік" інформації
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# ── Naive Bayes (Наївний Баєс) з BoW ──────────────────────────────────────────
# MultinomialNB — версія Наївного Баєса для цілих чисел (підходить для BoW)
# Ідея: обчислює ймовірність кожної категорії на основі частоти слів
# "Наївний" — бо вважає всі слова незалежними одне від одного (спрощення)
nb = MultinomialNB()
nb.fit(X_train_bow, y_train)           # навчання на тренувальних даних
pred_nb = nb.predict(X_test_bow)       # передбачення на тестових даних
acc_nb = accuracy_score(y_test, pred_nb)  # точність = частка правильних відповідей

# ── Random Forest (Випадковий ліс) з BoW ──────────────────────────────────────
# Будує багато дерев рішень і об'єднує їх голосуванням
# Більш потужний ніж Naive Bayes, але повільніший
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_bow, y_train)
pred_rf = rf.predict(X_test_bow)
acc_rf = accuracy_score(y_test, pred_rf)


# ── Крок 3: Модель FastText — отримання векторів документів ───────────────────

# FastText (схожий на Word2Vec, але краще обробляє незнайомі слова)
# потребує список списків токенів, а не рядки
# word_tokenize("Hello world!") → ["Hello", "world", "!"]
# .lower() — все в нижній регістр для однаковості
tokenized_train = [word_tokenize(text.lower()) for text in X_train]
tokenized_test = [word_tokenize(text.lower()) for text in X_test]

# Навчаємо FastText тільки на тренувальних даних
# vector_size=100 — кожне слово = вектор з 100 чисел
# window=5        — контекст: 5 сусідніх слів зліва і справа
# min_count=1     — включати навіть слова що зустрічаються 1 раз
ft_model = FastText(sentences=tokenized_train, vector_size=100, window=5, min_count=1)


def document_vector(tokens, model):
    """
    Перетворює список токенів (слів) одного документа на один вектор.
    
    Ідея: беремо вектор кожного слова і усереднюємо їх.
    Результат — один вектор з 100 чисел що представляє весь документ.
    
    Аргументи:
      tokens — список слів документа, наприклад ["fix", "laptop", "screen"]
      model  — навчена FastText модель
    
    Повертає: numpy масив з 100 чисел (середній вектор документа)
    """
    # Беремо вектори тільки для слів що є в словнику моделі
    vectors = [model.wv[word] for word in tokens if word in model.wv]

    # Якщо жодне слово не знайдено — повертаємо нульовий вектор
    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    # np.mean(axis=0) — усереднюємо по стовпцях (поелементне середнє)
    # Тобто кожна з 100 позицій вектора = середнє значення цієї позиції по всіх словах
    return np.mean(vectors, axis=0)


# Перетворюємо кожен документ на один вектор з 100 чисел
# np.array([ ... ]) — створюємо матрицю (кількість_документів × 100)
X_train_ft = np.array([document_vector(tokens, ft_model) for tokens in tokenized_train])
X_test_ft = np.array([document_vector(tokens, ft_model) for tokens in tokenized_test])

# ── Naive Bayes з FastText ─────────────────────────────────────────────────────
# GaussianNB — версія Наївного Баєса для дійсних чисел (підходить для FastText векторів)
# (MultinomialNB не підійшов би бо FastText дає дробові числа, в т.ч. від'ємні)
nb_ft = GaussianNB()
nb_ft.fit(X_train_ft, y_train)
pred_nb_ft = nb_ft.predict(X_test_ft)
acc_nb_ft = accuracy_score(y_test, pred_nb_ft)

# ── Random Forest з FastText ───────────────────────────────────────────────────
rf_ft = RandomForestClassifier(random_state=42)
rf_ft.fit(X_train_ft, y_train)
pred_rf_ft = rf_ft.predict(X_test_ft)
acc_rf_ft = accuracy_score(y_test, pred_rf_ft)


# ── Крок 4: GridSearch — автоматичний підбір найкращих параметрів ─────────────
# GridSearch перебирає всі комбінації параметрів і вибирає ту що дає найкращу точність
# cv=3 — крос-валідація з 3 фолдами (ділить тренувальні дані на 3 частини і тестує по черзі)
# n_jobs=-1 — використовувати всі ядра процесора для прискорення

# ── GridSearch: BoW + Naive Bayes ─────────────────────────────────────────────
# Pipeline об'єднує кілька кроків в один об'єкт
# Зручно: GridSearch підбирає параметри і для векторизатора і для моделі разом
pipeline_nb = Pipeline([
    ('vect', CountVectorizer()),  # крок 1: текст → вектор чисел
    ('nb', MultinomialNB())       # крок 2: вектор → передбачення категорії
])

# Параметри для перебору:
# vect__ngram_range — (1,1) = окремі слова; (1,2) = також пари слів ("network error")
# vect__max_features — скільки найчастіших слів брати (None = всі)
# nb__alpha — згладжування Лапласа: захищає від нульових ймовірностей
param_grid_nb = {
    'vect__ngram_range': [(1,1), (1,2)],
    'vect__max_features': [None, 5000, 10000],
    'nb__alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
}

print("Шукаємо найкращі параметри для BoW + Naive Bayes...")
grid_nb = GridSearchCV(pipeline_nb, param_grid_nb, cv=3, n_jobs=-1)
grid_nb.fit(X_train, y_train)           # перебираємо всі комбінації на тренувальних даних
best_nb_acc = grid_nb.score(X_test, y_test)  # оцінюємо найкращу комбінацію на тестових

# ── GridSearch: BoW + Random Forest ───────────────────────────────────────────
pipeline_rf = Pipeline([
    ('vect', CountVectorizer()),
    ('rf', RandomForestClassifier(random_state=42))
])

# Параметри для перебору:
# n_estimators    — кількість дерев у лісі (більше = точніше, але повільніше)
# max_depth       — максимальна глибина дерева (None = без обмежень)
# min_samples_split — мінімальна кількість зразків для розгалуження вузла
param_grid_rf = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 20, 50],
    'rf__min_samples_split': [2, 5]
}

print("Шукаємо найкращі параметри для BoW + Random Forest...")
grid_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=3, n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf_acc = grid_rf.score(X_test, y_test)

# ── GridSearch: FastText + Naive Bayes ────────────────────────────────────────
# var_smoothing — параметр згладжування для GaussianNB
# Запобігає нульовим дисперсіям (коли всі значення ознаки однакові)
param_grid_nb_ft = {
    'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
}

print("Шукаємо найкращі параметри для FastText + Naive Bayes...")
grid_nb_ft = GridSearchCV(GaussianNB(), param_grid_nb_ft, cv=3, n_jobs=-1)
grid_nb_ft.fit(X_train_ft, y_train)
best_nb_ft_acc = grid_nb_ft.score(X_test_ft, y_test)

# ── GridSearch: FastText + Random Forest ──────────────────────────────────────
param_grid_rf_ft = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 20, 50],
    'min_samples_split': [2, 5]
}

print("Шукаємо найкращі параметри для FastText + Random Forest...")
grid_rf_ft = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf_ft, cv=3, n_jobs=-1)
grid_rf_ft.fit(X_train_ft, y_train)
best_rf_ft_acc = grid_rf_ft.score(X_test_ft, y_test)


# ── Крок 5: Вивід результатів ─────────────────────────────────────────────────
# Порівнюємо точність (accuracy) до і після підбору параметрів
# Accuracy = кількість правильних передбачень / загальна кількість тестових прикладів

print("\n" + "="*40)
print("=== ДО GridSearch ===")
print("="*40)
print(f"BoW + Naive Bayes:          {acc_nb:.5f}")
print(f"BoW + Random Forest:        {acc_rf:.5f}")
print(f"FastText + Naive Bayes:     {acc_nb_ft:.5f}")
print(f"FastText + Random Forest:   {acc_rf_ft:.5f}")

print("\n" + "="*40)
print("=== ПІСЛЯ GridSearch ===")
print("="*40)
print(f"BoW + Naive Bayes (tuned):        {best_nb_acc:.5f}")
print(f"BoW + Random Forest (tuned):      {best_rf_acc:.5f}")
print(f"FastText + Naive Bayes (tuned):   {best_nb_ft_acc:.5f}")
print(f"FastText + Random Forest (tuned): {best_rf_ft_acc:.5f}")
