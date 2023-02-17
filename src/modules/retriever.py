import unicodedata
import csv
import random
from drqa import retriever


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


class TfidfRetriever():
    """
    The retriever is built with the codebase: https://github.com/efficientqa/retrieval-based-baselines#tfidf-retrieval
    """
    def __init__(self, db_path, ranker_index):
        self.db, self.ranker = self.load_retriever(db_path, ranker_index)

    def load_retriever(self, db_path, ranker_index):
        print("Loading NQ train database")
        db = {}
        with open(db_path) as f:
            db_data = csv.reader(f, delimiter="\t")
            
            for line in db_data:
                if line[0] == "id":
                    continue
                db[line[0]] = (line[1], line[2])


        print("Loading Ranker index")
        ranker = retriever.get_class('tfidf')(tfidf_path=ranker_index)

        return db, ranker

    def fetch_text(self, doc_id):
        return self.db[doc_id]

    def get_KB(self, text, topk=10):
        try:
            doc_names, doc_scores = self.ranker.closest_docs(text, k=topk)

            shots = []
            for doc_name, doc_score in zip(doc_names, doc_scores):
                para = self.fetch_text(doc_name)

                shots.append(para)
            return shots
        except:
            return []


class RandomRetriever():
    def __init__(self, db_path):
        self.db, self.ranker = self.load_retriever(db_path)

    def load_retriever(self, db_path):
        print("Loading NQ train database")
        db = {}
        with open(db_path) as f:
            db_data = csv.reader(f, delimiter="\t")
            
            for line in db_data:
                if line[0] == "id":
                    continue
                db[line[0]] = (line[1], line[2])


        print("Loading Ranker index")
        ranker = list(db.keys())

        return db, ranker

    def fetch_text(self, doc_id):
        return self.db[doc_id]

    def get_KB(self, seed, topk=10):
        random.seed(seed)
        doc_names = random.sample(self.ranker, topk)

        shots = []
        for doc_name in doc_names:
            para = self.fetch_text(doc_name)

            shots.append(para)
        return shots


if __name__ == "__main__":
    topk = 10

    tfidf_retriever = TfidfRetriever()
    results = tfidf_retriever.get_KB("what do the 3 dots mean in math", topk=topk)
    print(results)