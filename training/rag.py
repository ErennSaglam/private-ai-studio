"""Minimal local RAG: BM25 retrieval over the law Q&A answers + prompt construction.

No embedding model and no extra package: BM25 is a lexical ranking function (Robertson & Zaragoza, 2009).
It is the baseline; an embedding retriever can replace `BM25Index` later without touching the rest.

Corpus: the answer texts of all cleaned rows, EXCEPT the test rows themselves. Otherwise the model
could copy the reference answer and the test would measure retrieval of a known string, not RAG.

Usage (from the project root):
    python training/rag.py build                      # -> data/law-qa/rag_corpus.jsonl
    python training/rag.py search "kira bedeli ödenmezse ne olur?"
"""
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/turkish_law_qa/train-00000-of-00001.parquet"
DATA = ROOT / "data/law-qa"
CORPUS = DATA / "rag_corpus.jsonl"

# Very common Turkish function words; removing them makes BM25 focus on legal terms.
STOPWORDS = set(
    "ve veya ile bir bu şu o da de ki mi mı mu mü için gibi olarak olan olarak ise ne nedir nelerdir "
    "nasıl hangi hangileri kadar daha en çok her bazı tüm ancak fakat ama yani hem göre sonra önce "
    "durumda durumunda durumlarda olup olması olmak olduğu olduğunu tarafından ilgili şekilde".split()
)


def tr_lower(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("I", "ı").replace("İ", "i").lower()


def tokenize(text: str) -> list[str]:
    """Lower-case words without stopwords, cut to 6 characters.

    Turkish is agglutinative: "davasının", "davası", "davada" share the stem "dava".
    A fixed-length prefix is a crude but dependency-free stand-in for a stemmer.
    """
    return [w[:6] for w in re.findall(r"\w+", tr_lower(text)) if w not in STOPWORDS and len(w) > 1]


class BM25Index:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avgdl = sum(self.len) / len(self.len)
        df = Counter(term for t in self.tf for term in t)
        n = len(docs)
        # BM25 idf with +1 inside the log so it never becomes negative.
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, t in enumerate(self.tf):
            for term in t:
                self.postings.setdefault(term, []).append(i)

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        scores: Counter = Counter()
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                f = self.tf[i][term]
                norm = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / norm
        return scores.most_common(k)


def build_corpus() -> None:
    """Write the retrieval corpus: every cleaned answer except the test answers."""
    import pandas as pd  # only needed for building

    sys.path.insert(0, str(Path(__file__).parent))
    from prepare_data import META_PATTERN, key, normalize

    df = pd.read_parquet(RAW)
    df["question"], df["answer"] = df["question"].map(normalize), df["answer"].map(normalize)
    df["source_row"] = range(len(df))
    df = df[~(df["question"].str.contains(META_PATTERN) | df["answer"].str.contains(META_PATTERN))]
    df = df[~df["question"].map(key).duplicated()]

    test = pd.read_csv(DATA / "test.csv")
    test_answers = set(test["Answer"].map(key))
    corpus = df[~df["answer"].map(key).isin(test_answers)]
    with open(CORPUS, "w", encoding="utf-8") as f:
        for row in corpus.itertuples():
            f.write(json.dumps({"id": int(row.source_row), "text": row.answer}, ensure_ascii=False) + "\n")
    print(f"corpus: {len(corpus)} passages (removed {len(df) - len(corpus)} test answers) -> {CORPUS}")


class Retriever:
    def __init__(self, corpus_path: Path = CORPUS):
        self.passages = [json.loads(line) for line in open(corpus_path, encoding="utf-8")]
        self.index = BM25Index([p["text"] for p in self.passages])

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        return [{**self.passages[i], "score": round(s, 2)} for i, s in self.index.search(query, k)]


RAG_INSTRUCTION = (
    "Aşağıdaki kaynaklar soruyla ilgili olabilir. Cevabını yalnızca bu kaynaklara dayandır. "
    "Kaynaklarda cevap yoksa bunu açıkça söyle."
)


def rag_messages(messages: list[dict], passages: list[dict]) -> list[dict]:
    """Insert the retrieved passages into the system message, keep the question unchanged."""
    sources = "\n".join(f"[{n}] {p['text']}" for n, p in enumerate(passages, 1))
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    rest = messages[1:] if system else messages
    return [{"role": "system", "content": f"{system}\n\n{RAG_INSTRUCTION}\n\nKaynaklar:\n{sources}".strip()}, *rest]


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build_corpus()
    elif sys.argv[1:2] == ["search"]:
        for hit in Retriever().retrieve(" ".join(sys.argv[2:]), k=3):
            print(f"{hit['score']:6.2f}  [{hit['id']}] {hit['text'][:160]}")
    else:
        print(__doc__)
