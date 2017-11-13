
from nltk.tokenize import sent_tokenize, word_tokenize
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StandardAnalyzer
from whoosh.qparser import QueryParser
import os


def main():
    file_content_doc1 = open("science.txt").read()
    file_content_doc2 = open("rural.txt").read()
    sent_tokenize_list1 = sent_tokenize(file_content_doc1, language='english')
    sent_tokenize_list2 = sent_tokenize(file_content_doc2,language='english')

    if not os.path.exists("index_for_sample_files"):
        os.mkdir("index_for_sample_files")

    schema = Schema(full_text=TEXT(phrase=True, analyzer=StandardAnalyzer(stoplist=None)))
    ix = create_in("index_for_sample_files", schema)
    writer = ix.writer()

    for sentence in sent_tokenize_list1:
        writer.add_document(full_text=sentence)
    for sentence in sent_tokenize_list2:
        writer.add_document(full_text=sentence)
    writer.commit()

    with ix.searcher() as searcher:
       #print(list(searcher.lexicon("content")))
       query = QueryParser("full_text", ix.schema).parse("ploy by overseas interest to try and get the single desk put aside")
       results = searcher.search(query, limit = 10)



if __name__ == "__main__":
    main()