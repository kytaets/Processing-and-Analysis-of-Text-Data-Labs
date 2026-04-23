import pandas as pd
import numpy as np
import random
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def preprocess(text, stop_words, punctuation):
    tokens = word_tokenize(str(text).lower())
    return [word for word in tokens if word not in stop_words and word not in punctuation and len(word) > 2]

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=0)
        model_list.append(model)
        
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print(f"Кількість тем: {num_topics}, Оцінка узгодженості (C_v): {coherence_values[-1]:.4f}")

    return model_list, coherence_values

if __name__ == '__main__':
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

    print("Зчитування даних...")
    file_path = 'ecommerceDataset4.csv' 
    df = pd.read_csv(file_path)

    text_column = 'text'   
    class_column = 'category' 

    df_sampled = df.groupby(class_column, group_keys=False).apply(
        lambda x: x.sample(frac=0.2, random_state=42), 
        include_groups=False
    )

    documents = df_sampled[text_column].dropna().tolist()
    print(f"Відібрано документів для аналізу: {len(documents)}")

    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)

    print("Попередня обробка документів...")
    processed_docs = [preprocess(doc, stop_words, punctuation) for doc in documents if len(str(doc)) > 10]

    dictionary = corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    print("\nПошук оптимальної кількості тем...")
    model_list, coherence_values = compute_coherence_values(
        dictionary=dictionary, corpus=bow_corpus, texts=processed_docs, start=2, limit=10, step=2
    )

    optimal_index = np.argmax(coherence_values)
    best_lda_model = model_list[optimal_index]
    optimal_num_topics = best_lda_model.num_topics
    print(f"\nОптимальна кількість тем: {optimal_num_topics}")

    print("\nТерми для кожної теми:")
    topics = best_lda_model.print_topics(num_words=5)
    for topic in topics:
        print(topic)

    print("\nАналіз трьох випадкових документів:")
    random_indices = random.sample(range(len(bow_corpus)), 3)

    for idx in random_indices:
        doc_topics = best_lda_model.get_document_topics(bow_corpus[idx])
        doc_topics.sort(key=lambda x: x[1], reverse=True)
        most_important_topic = doc_topics[0][0]
        topic_probability = doc_topics[0][1]
        
        print(f"\nДокумент #{idx} (перші 50 символів): '{' '.join(processed_docs[idx][:10])}...'")
        print(f"Найважливіша тема: {most_important_topic} (Ймовірність: {topic_probability:.4f})")