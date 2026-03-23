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

nltk.download('punkt', quiet=True) 

df = pd.read_csv("IT_tickets2.csv")

df = df.dropna(subset=['Document', 'Topic_group'])

texts = df['Document'].astype(str)
labels = df['Topic_group']

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_bow, y_train)
pred_nb = nb.predict(X_test_bow)
acc_nb = accuracy_score(y_test, pred_nb)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_bow, y_train)
pred_rf = rf.predict(X_test_bow)
acc_rf = accuracy_score(y_test, pred_rf)

tokenized_train = [word_tokenize(text.lower()) for text in X_train]
tokenized_test = [word_tokenize(text.lower()) for text in X_test]

ft_model = FastText(sentences=tokenized_train, vector_size=100, window=5, min_count=1)

def document_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

X_train_ft = np.array([document_vector(tokens, ft_model) for tokens in tokenized_train])
X_test_ft = np.array([document_vector(tokens, ft_model) for tokens in tokenized_test])

nb_ft = GaussianNB()
nb_ft.fit(X_train_ft, y_train)
pred_nb_ft = nb_ft.predict(X_test_ft)
acc_nb_ft = accuracy_score(y_test, pred_nb_ft)

rf_ft = RandomForestClassifier(random_state=42)
rf_ft.fit(X_train_ft, y_train)
pred_rf_ft = rf_ft.predict(X_test_ft)
acc_rf_ft = accuracy_score(y_test, pred_rf_ft)

pipeline_nb = Pipeline([
    ('vect', CountVectorizer()),
    ('nb', MultinomialNB())
])

param_grid_nb = {
    'vect__ngram_range': [(1,1), (1,2)],
    'vect__max_features': [None, 5000, 10000],
    'nb__alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
}

print("Шукаємо найкращі параметри для BoW + Naive Bayes...")
grid_nb = GridSearchCV(pipeline_nb, param_grid_nb, cv=3, n_jobs=-1)
grid_nb.fit(X_train, y_train)
best_nb_acc = grid_nb.score(X_test, y_test)

pipeline_rf = Pipeline([
    ('vect', CountVectorizer()),
    ('rf', RandomForestClassifier(random_state=42)) # Додано random_state
])

param_grid_rf = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 20, 50],
    'rf__min_samples_split': [2, 5]
}

print("Шукаємо найкращі параметри для BoW + Random Forest...")
grid_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=3, n_jobs=-1) # n_jobs=-1 прискорить процес
grid_rf.fit(X_train, y_train)
best_rf_acc = grid_rf.score(X_test, y_test)

param_grid_nb_ft = {
    'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
}

print("Шукаємо найкращі параметри для FastText + Naive Bayes...")
grid_nb_ft = GridSearchCV(GaussianNB(), param_grid_nb_ft, cv=3, n_jobs=-1)
grid_nb_ft.fit(X_train_ft, y_train)
best_nb_ft_acc = grid_nb_ft.score(X_test_ft, y_test)

param_grid_rf_ft = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 20, 50],
    'min_samples_split': [2, 5]
}

print("Шукаємо найкращі параметри для FastText + Random Forest...")
grid_rf_ft = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf_ft, cv=3, n_jobs=-1)
grid_rf_ft.fit(X_train_ft, y_train)
best_rf_ft_acc = grid_rf_ft.score(X_test_ft, y_test)

print("\n" + "="*40)
print("=== ДО GridSearch ===")
print("="*40)
print(f"BoW + Naive Bayes: {acc_nb:.5f}")
print(f"BoW + Random Forest: {acc_rf:.5f}")
print(f"FastText + Naive Bayes: {acc_nb_ft:.5f}")
print(f"FastText + Random Forest: {acc_rf_ft:.5f}")

print("\n" + "="*40)
print("=== ПІСЛЯ GridSearch ===")
print("="*40)
print(f"BoW + Naive Bayes (tuned): {best_nb_acc:.5f}")
print(f"BoW + Random Forest (tuned): {best_rf_acc:.5f}")
print(f"FastText + Naive Bayes (tuned): {best_nb_ft_acc:.5f}")
print(f"FastText + Random Forest (tuned): {best_rf_ft_acc:.5f}")
