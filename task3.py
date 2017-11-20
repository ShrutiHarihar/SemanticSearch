from whoosh.analysis import StemFilter, Filter, LowercaseFilter, StopFilter, RegexTokenizer
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in
from whoosh.qparser import QueryParser
from whoosh import qparser, query, sorting
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import os
import itertools

def main():
    file_content_doc1 = open("science.txt").read()
    file_content_doc2 = open("rural.txt").read()
    sent_tokenize_list1 = sent_tokenize(file_content_doc1, language='english')
    sent_tokenize_list2 = sent_tokenize(file_content_doc2, language='english')

    if not os.path.exists("index_for_sample_files_task3"):
        os.mkdir("index_for_sample_files_task3")
    my_analyzer = RegexTokenizer()| StopFilter()| LowercaseFilter() | StemFilter() | PosTagger()| Lemmatizer()
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
        og = qparser.OrGroup.factory(0.9)
        query_text = QueryParser("stem_text", schema = ix.schema, group= og).parse(
            "who is controlling the threat of locusts?")
        print(query_text)
        results = searcher.search(query_text, sortedby= scores, limit = 10 )
        for hit in results:
            print(hit["stem_text"])


#filter for lemmatizing the data
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
                t.text = lemma
                yield t

#filter for Pos Tagging
class PosTagger(Filter):
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
         words = []
         tokens1, tokens2 = itertools.tee(tokens)
         for t in tokens1:
            words.append(t.text)
         tags = pos_tag(words)
         i=0
         for t in tokens2:
             t.text = tags[i][0] + " "+ tags[i][1]
             i += 1
             yield t


if __name__ == "__main__":
    main()

