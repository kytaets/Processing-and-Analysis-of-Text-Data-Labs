import re
from pathlib import Path
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

INPUT_PATH = Path("text11.txt")

PHONE_RE = re.compile(r"""
(?:\+?\d[\d\-\s\(\)]{7,}\d)
""", re.VERBOSE)


def read_text_smart(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1251", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            pass
    return path.read_text(encoding="utf-8", errors="replace")


def download_nltk() -> None:
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg)

    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg)

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger_eng")
        except Exception:
            nltk.download("averaged_perceptron_tagger")


def get_wordnet_pos(treebank_tag: str):
    """Map Penn Treebank POS tags to WordNet POS tags for lemmatization."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return None


def print_changed_only(words: list[str], tagged: list[tuple[str, str]], lemmas: list[str]) -> None:
    print("\nChanged tokens (token -> lemma):")
    changed = False
    for (w, t), lemma in zip(tagged, lemmas):
        if w != lemma:
            changed = True
            print(f"{w} ({t}) -> {lemma}")
    if not changed:
        print("(No changes for this sentence.)")


def main():
    download_nltk()

    print("RUNNING FILE:", __file__)
    print("CWD:", os.getcwd())

    text = read_text_smart(INPUT_PATH)

    text = text.replace("�", " ")

    sentences = sent_tokenize(text)
    print("\n===== Sentences =====")
    print("Number of sentences:", len(sentences))

    if sentences:
        print("Last sentence:")
        print(sentences[-1])

    text_no_phones = PHONE_RE.sub(" ", text)
    tokens = [w for w in word_tokenize(text_no_phones) if w.isalpha()]

    print("\n===== Tokens without phone numbers (words only) =====")
    print(tokens)

    stop_words = set(stopwords.words("english"))
    tokens_no_stop = [w for w in tokens if w.lower() not in stop_words]

    print("\n===== Without stopwords =====")
    print(tokens_no_stop)

    if len(sentences) >= 2:
        lemmatizer = WordNetLemmatizer()

        penultimate = sentences[-2]
        penultimate = PHONE_RE.sub(" ", penultimate)

        words = [w for w in word_tokenize(penultimate) if w.isalpha()]
        tagged = pos_tag(words)

        lemmas = []
        for w, t in tagged:
            wn_pos = get_wordnet_pos(t)
            if wn_pos is None:
                lemmas.append(w)  
            else:
                lemmas.append(lemmatizer.lemmatize(w, wn_pos))
       
        print("\n===== Penultimate sentence =====")
        print(penultimate)

        print("\nTokens:")
        print(words)

        # print("\nPOS tags:")
        # print(tagged)

        print("\nLemmas:")
        print(lemmas)

        print_changed_only(words, tagged, lemmas)

    else:
        print("\nThere is no penultimate sentence in the text.")


if __name__ == "__main__":
    main()
