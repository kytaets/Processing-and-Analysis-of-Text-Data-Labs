import nltk
from nltk.corpus import gutenberg                                        # колекція класичних книг
from nltk.collocations import TrigramCollocationFinder, TrigramAssocMeasures  # пошук триграм

# Завантажуємо корпус Гутенберга — збірка класичних літературних творів у відкритому доступі
# Містить твори Шекспіра, Честертона, Мілтона та інших
nltk.download('gutenberg')

# Завантажуємо слова з книги Честертона "The Man Who Was Thursday"
# gutenberg.words() повертає одразу список токенів (слів і розділових знаків)
# Наприклад: ['[', 'The', 'Man', 'Who', 'Was', 'Thursday', 'by', ...]
words = gutenberg.words('chesterton-thursday.txt')

# ── Пошук триграм ─────────────────────────────────────────────────────────────
# Триграма — послідовність з ТРЬОХ слів що йдуть поруч у тексті
# Наприклад: "to be or", "the quick brown", "man who was"
#
# TrigramAssocMeasures — набір статистичних мір для оцінки "важливості" триграми
# (PMI, likelihood ratio, chi-square тощо)
trigram_measures = TrigramAssocMeasures()

# TrigramCollocationFinder.from_words() — сканує весь текст і
# знаходить всі можливі триграми та підраховує їх частоти
finder = TrigramCollocationFinder.from_words(words)

# ── Фільтрація ────────────────────────────────────────────────────────────────
# Фільтр 1: залишаємо тільки слова що складаються виключно з літер
# w.isalpha() → True для "hello", False для "123", "Mr.", ","
# Це прибирає розділові знаки, цифри, скорочення
finder.apply_word_filter(lambda w: not w.isalpha())

# Фільтр 2: залишаємо триграми що зустрічаються хоча б 3 рази
# Рідкісні триграми (1-2 рази) — можуть бути випадковими збігами
finder.apply_freq_filter(3)

# ── Ранжування за PMI ─────────────────────────────────────────────────────────
# PMI (Pointwise Mutual Information) — міра того, наскільки слова
# "притягуються" одне до одного порівняно з випадковим збігом.
#
# Висока PMI = слова зустрічаються разом НАБАГАТО частіше ніж по-окремо
# → це справжня стійка фраза (колокація)
#
# Формула: PMI(a,b,c) = log( P(a,b,c) / (P(a) * P(b) * P(c)) )
# Якщо слова незалежні → PMI ≈ 0
# Якщо слова завжди разом → PMI >> 0
#
# nbest(міра, n) — повертає топ-n триграм за вказаною мірою
print("Найкращі ключові триграми (за PMI):")
top_trigrams = finder.nbest(trigram_measures.pmi, 10)

# Виводимо 10 найбільш "зв'язних" триграм
# ' '.join(trigram) склеює три слова пробілом: ("the", "man", "who") → "the man who"
for idx, trigram in enumerate(top_trigrams, 1):
    print(f"{idx}. {' '.join(trigram)}")
