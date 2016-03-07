#!/usr/bin/python
from operator import itemgetter

from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

sample_corpus = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey"
]


class Document(object):

    def __init__(self, id, body):
        self.id = id
        self.body = body

    def __repr__(self):

        return '{}: [{}...]'.format(
            self.id,
            self.body[0:20]
        )


def read_corpus_from_file(filepath):

    id_mark = '.I'
    body_mark = '.W'

    documents = []

    with open(filepath, 'r') as fp:

        sentences = []
        document_id = None

        for line in fp:

            if line.startswith(id_mark):

                # save previous doc
                if document_id:

                    document = Document(document_id, '\n'.join(sentences))
                    documents.append(document)

                document_id = int(line.replace(id_mark, '').strip())
                sentences = []
            elif line.startswith(body_mark):
                pass
            else:
                sentences.append(line)


def preprocess_document(doc):
    """
    [tokens]
    :param doc:
    :return:
    """
    stopset = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = wordpunct_tokenize(doc)
    clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2]
    final = [stemmer.stem(word) for word in clean]
    return final


def create_dictionary(docs):
    """
    {term: frequency}
    :param docs:
    :return:
    """

    pdocs = [preprocess_document(doc) for doc in docs]
    dictionary = corpora.Dictionary(pdocs)
    dictionary.save('/tmp/vsm.dict')
    return dictionary


def get_keyword_to_id_mapping(dictionary):
    print dictionary.token2id


def docs2bows(corpus, dictionary):
    """
    (termId, frequency)
    :param corpus:
    :param dictionary:
    :return:
    """
    docs = [preprocess_document(d) for d in corpus]
    vectors = [dictionary.doc2bow(doc) for doc in docs]
    corpora.MmCorpus.serialize('/tmp/vsm_docs.mm', vectors)
    return vectors


def create_TF_IDF_model(corpus):
    """

    :param corpus:
    :return:
    """

    dictionary = create_dictionary(corpus)
    docs2bows(corpus, dictionary)
    loaded_corpus = corpora.MmCorpus('/tmp/vsm_docs.mm')
    tfidf = models.TfidfModel(loaded_corpus)
    return tfidf, dictionary


def launch_query(corpus, q):
    """

    :param corpus:
    :param q:
    :return:
    """

    tfidf, dictionary = create_TF_IDF_model(corpus)
    loaded_corpus = corpora.MmCorpus('/tmp/vsm_docs.mm')
    index = similarities.MatrixSimilarity(loaded_corpus, num_features=len(dictionary))

    pq = preprocess_document(q)
    vq = dictionary.doc2bow(pq)
    print 'PQ: {}'.format(pq)
    print 'VQ: {}'.format(vq)
    qtfidf = tfidf[vq]
    sim = index[qtfidf]
    ranking = sorted(enumerate(sim), key=itemgetter(1), reverse=True)
    for doc, score in ranking:
        print "[ Score = " + "%.3f" % round(score, 3) + "] " + corpus[doc]


if __name__ == '__main__':

    launch_query(sample_corpus, 'path')


