from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    BM25_K1,
    load_movies, 
    load_stopwords)
import math
from nltk.stem import PorterStemmer
import os
import pickle
import string
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_PATH, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_PATH, "docmap.pkl")

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_PATH, "term_frequencies.pkl")
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term):
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            self.docmap[movie['id']] = movie
            self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")

    def load(self):
        with open(INDEX_PATH, 'rb') as index_file:
            self.index = pickle.load(index_file)
        with open(DOCMAP_PATH, 'rb') as docmap_file:
            self.docmap = pickle.load(docmap_file)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

    def save(self):
        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)
        with open(INDEX_PATH, "wb") as index_file:
            pickle.dump(self.index, index_file)
        with open(DOCMAP_PATH, "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]
    
    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log(((doc_count - term_doc_count) + 0.5) / (term_doc_count + 0.5) +1)
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1) -> float:
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> set:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def filter_words(tokens: list[str], filter_words: list[str]) -> list[str]:
    for word in filter_words:
        if word in tokens:
            tokens.remove(word)
    return tokens


def stem_tokens(tokens: list[str]) -> list[str]:
    word_stems = []
    stemmer = PorterStemmer()
    for token in tokens:
        word_stems.append(stemmer.stem(token))
    return word_stems


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    
    stop_words = load_stopwords()
    filtered_tokens = filter_words(valid_tokens, stop_words)
    
    stemmed_tokens = stem_tokens(filtered_tokens)

    return stemmed_tokens


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)


def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    tfidf = idx.get_tf(doc_id, term) * idx.get_idf(term)
    return tfidf


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    bm25idf = idx.get_bm25_idf(term)
    return bm25idf


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1) -> float:
    idx = InvertedIndex()
    idx.load()
    bm25_tf = idx.get_bm25_tf(doc_id, term, k1)
    return bm25_tf
    