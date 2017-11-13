from whoosh.analysis import StandardAnalyzer, StemmingAnalyzer
from whoosh.fields import Schema, TEXT, ID
from nltk.tokenize import sent_tokenize
from whoosh.index import create_in
from whoosh.qparser import QueryParser
from whoosh import qparser, query
from whoosh import columns, fields, index, sorting
import codecs, sys, glob, os, unicodedata

def main():
    file_content_doc1 = open("science.txt").read()
    file_content_doc2 = open("rural.txt").read()
    sent_tokenize_list1 = sent_tokenize(file_content_doc1, language='english')
    sent_tokenize_list2 = sent_tokenize(file_content_doc2, language='english')
    
    if not os.path.exists("index_for_sample_files_task3"):
        os.mkdir("index_for_sample_files_task3")
    schema = Schema(id=ID(stored=True, unique=True), stem_text=TEXT(stored=True,analyzer=StemmingAnalyzer()))
    ix = create_in("index_for_sample_files_task3", schema)
    writer = ix.writer()

    for sentence in sent_tokenize_list1:
        writer.add_document(stem_text=sentence)
    for sentence in sent_tokenize_list2:
        writer.add_document(stem_text=sentence)
    writer.commit()

    scores = sorting.ScoreFacet()
    with ix.searcher() as searcher:
        query_text = QueryParser("stem_text", ix.schema, termclass=query.Variations, group= qparser.OrGroup).parse(
            "astonish crowd")
        results = searcher.search(query_text, limit=10, sortedby=scores)
        for i in range(10):
            print(results[i])




if __name__ == "__main__":
    main()

