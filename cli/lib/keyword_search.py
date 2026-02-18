from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies
import string


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    preprocessed_query = preprocess_text(query)
    for movie in movies:
        preprocessed_title = preprocess_text(movie["title"])
        
        for word in preprocessed_query:
            if word in " ".join(preprocessed_title):
                results.append(movie)
                break
        if len(results) >= limit:
            break

        '''
        if preprocessed_query.intersection(preprocessed_title):
            results.append(movie)
            if len(results) >= limit:
                break
        '''
    return results


def preprocess_text(text: str) -> set:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    textset = set(filter(None, text.split()))
    return textset