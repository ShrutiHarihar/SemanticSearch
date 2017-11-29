from whoosh.analysis import StandardAnalyzer, StemmingAnalyzer, Filter, LowercaseFilter, StopFilter, RegexTokenizer, Tokenizer, Token
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in
from whoosh.qparser import  MultifieldParser
from whoosh import qparser
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from whoosh.scoring import BM25F
from nltk import pos_tag
import os, spacy
import itertools
import whoosh

def main():
    file_content_doc1 = open("science.txt").read()
    file_content_doc2 = open("rural.txt").read()
    sent_tokenize_list1 = sent_tokenize(file_content_doc1, language='english')
    sent_tokenize_list2 = sent_tokenize(file_content_doc2, language='english')

    if not os.path.exists("index_for_sample_files_task3_min"):
        os.mkdir("index_for_sample_files_task3_min")

    my_analyzer = RegexTokenizer()| StopFilter()| LowercaseFilter() | Lemmatizer()
    pos_tagger = RegexTokenizer()| StopFilter()| LowercaseFilter()| PosTagger()
    schema = Schema(id=ID(stored=True, unique=True),standard = TEXT(stored= True, analyzer= StandardAnalyzer()),  stem_text=TEXT( analyzer= StemmingAnalyzer()), lemma = TEXT( analyzer=my_analyzer), pos_text=TEXT( analyzer=pos_tagger), headword=TEXT(analyzer=DependencyParser()))
    ix = create_in("index_for_sample_files_task3_min", schema)
    writer = ix.writer()

    for sentence in sent_tokenize_list1:
        writer.add_document(standard = sentence,stem_text = sentence, lemma = sentence, pos_text= sentence, headword = sentence)
    for sentence in sent_tokenize_list2:
        writer.add_document(standard = sentence,stem_text = sentence, lemma = sentence, pos_text= sentence, headword = sentence)
    writer.commit()
    with ix.searcher(weighting=whoosh.scoring.BM25F()) as searcher:
        og = qparser.OrGroup.factory(0.5)
        query_text = MultifieldParser(["headword","standard","stem_text","lemma","pos_text"], schema = ix.schema, group = og).parse(
            "what will reduce cystic fibrosis?")
        #print(query_text)
        results = searcher.search(query_text, limit = 10)
        for i, hit in enumerate(results):
            print(results.score(i), hit["standard"], sep=":" )
            print("\n")

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

class DependencyParser(Tokenizer):
    def __eq__(self, other):
        return other and self.__class__ is other.__class__

    def __call__(self, value, positions=False, chars=False, keeporiginal=False,
                     removestops=True, start_pos=0, start_char=0, tokenize=True,
                     mode='', **kwargs):
        t = Token(positions, chars, removestops=removestops, mode=mode,
                      **kwargs)
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(value)
        t.pos = start_pos
        for chunk in doc.noun_chunks:
            t.text = chunk.root.dep_
            yield t


if __name__ == "__main__":
    main()

