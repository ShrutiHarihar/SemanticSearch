from whoosh.analysis import StemFilter, Filter, LowercaseFilter, StopFilter, RegexTokenizer
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in
from whoosh.qparser import QueryParser
from whoosh import qparser, query, sorting
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import os

def main():
    file_content_doc1 = open("science.txt").read()
    file_content_doc2 = open("rural.txt").read()
    sent_tokenize_list1 = sent_tokenize(file_content_doc1, language='english')
    sent_tokenize_list2 = sent_tokenize(file_content_doc2, language='english')

    if not os.path.exists("index_for_sample_files_task3"):
        os.mkdir("index_for_sample_files_task3")
    my_analyzer = RegexTokenizer()| StopFilter()| LowercaseFilter() | StemFilter()  | Lemmatizer()
    schema = Schema(id=ID(stored=True, unique=True), stem_text=TEXT(stored= True, analyzer=my_analyzer))
    ix = create_in("index_for_sample_files_task3", schema)
    writer = ix.writer()

    for sentence in sent_tokenize_list1:
        writer.add_document(stem_text = sentence)
    for sentence in sent_tokenize_list2:
        writer.add_document(stem_text = sentence)
    writer.commit()

    scores = sorting.ScoreFacet()

    with ix.searcher() as searcher:
        query_text = QueryParser("stem_text", ix.schema, termclass=query.Variations, group= qparser.OrGroup).parse(
            "what is something pretty astonishing?")
        results = searcher.search(query_text, sortedby= scores, limit = 10)
        for hit in results:
            print(hit["stem_text"])



class Lemmatizer(Filter):
    def __eq__(self, other):
        return (other
                and self.__class__ is other.__class__
                and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self == other

    def __init__(self):
        self.cache = {}

    def __call__(self, tokens):
        assert hasattr(tokens, "__iter__")
        lm = WordNetLemmatizer()
        for t in tokens:
            if t.stopped:
                yield t
                continue
            text = t.text
            if text in self.cache:
                t.text = self.cache[text]
                yield t
            else:
                lemma = lm.lemmatize(text)
                self.cache[t.text] = lemma
                yield t


if __name__ == "__main__":
    main()

