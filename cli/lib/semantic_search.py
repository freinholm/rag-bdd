from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')


def verify_model() -> None:
    s = SemanticSearch()
    print(f"Model loaded: {s.model}")
    print(f"Max sequence length: {s.model.max_seq_length}")
