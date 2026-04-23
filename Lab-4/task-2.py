import nltk
from nltk.corpus import gutenberg
from nltk.collocations import TrigramCollocationFinder, TrigramAssocMeasures

nltk.download('gutenberg')

words = gutenberg.words('chesterton-thursday.txt')

trigram_measures = TrigramAssocMeasures()
finder = TrigramCollocationFinder.from_words(words)

finder.apply_word_filter(lambda w: not w.isalpha())
finder.apply_freq_filter(3)

print("Найкращі ключові триграми (за PMI):")
top_trigrams = finder.nbest(trigram_measures.pmi, 10)

for idx, trigram in enumerate(top_trigrams, 1):
    print(f"{idx}. {' '.join(trigram)}")