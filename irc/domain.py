import collections
import operator

import gensim
import nltk.corpus
import nltk.stem
import nltk.tokenize
import pandas


def pre_process_document(doc, lang='english'):
    stopset = set(nltk.corpus.stopwords.words(lang))
    stemmer = nltk.stem.PorterStemmer()
    tokens = nltk.tokenize.wordpunct_tokenize(doc)
    clean = (token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2)
    final = [stemmer.stem(word) for word in clean]
    return final


def create_dictionary(corpus):
    pdocs = (pre_process_document(doc.body) for doc in corpus.docs)
    dictionary = gensim.corpora.Dictionary(pdocs)

    return dictionary


def create_bag_of_words(corpus, dictionary, normalizer=None):
    docs = (pre_process_document(d.body) for d in corpus.docs)
    vectors = [dictionary.doc2bow(doc) for doc in docs]

    if normalizer:
        def normalize_vector(v, norm=normalizer):
            return zip((x[0] for x in v), map(norm, (x[1] for x in v)))

        vectors = map(normalize_vector, vectors)

    return vectors


class Document(object):
    def __init__(self, id, body):
        self.id = id
        self.body = body

    def __repr__(self):
        return "{}({}: '{}...')".format(
            self.__class__.__name__,
            self.id,
            self.body[0:20]
        )


class Corpus(object):
    def __init__(self, docs):
        assert isinstance(docs, list)
        self.docs = docs


class Query(object):
    def __init__(self, id, term):
        self.id = id
        self.term = term
        self.results = set()
        self.relevant_docs = set()
        self.evaluation = pandas.DataFrame(columns=['precision', 'recall'])
        self.num_docs_retrieved = None
        self.num_relevant_docs_retrieved = None
        self.precision = None
        self.recall = None

    def __repr__(self):
        return "{}({}: '{}...')".format(
            self.__class__.__name__,
            self.id,
            self.term[0:20]
        )


class Index(object):
    def __init__(self, corpus):
        self._corpus = corpus
        self._dictionary = None
        self._bag_of_words = None
        self._model = None
        self.index = None

        self.create_model(corpus)

    def create_model(self, corpus):
        raise NotImplementedError

    @property
    def model(self):
        raise NotImplementedError

    def search(self, q, n=None, feedback=None):
        raise NotImplementedError

    @property
    def corpus(self):
        return self._corpus


class BinaryIndex(Index):
    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, len(self._bag_of_words))

    def create_model(self, corpus):
        self._dictionary = create_dictionary(self._corpus)
        self._bag_of_words = create_bag_of_words(self._corpus, self._dictionary, normalizer=binary_tf)

    @property
    def model(self):
        return self._model

    def search(self, q, n=None, feedback=None):

        pq = pre_process_document(q)

        if feedback:
            _docs_term = {}
            _chosen_terms = []
            _counter = collections.Counter()

            for doc in self.corpus.docs:
                if doc.id in feedback:
                    _docs_term[doc.id] = set(pre_process_document(doc.body))
                    for _term in _docs_term[doc.id]:
                        _counter[_term] += 1

            for _term, _count in _counter.items():
                if _count >= len(feedback) / 2.0 and _term not in pq:
                    _chosen_terms.append(_term)
                    pq.append(_term)

            print('Using feedback terms: {}...'.format(_chosen_terms[0:10]))

        vq = self._dictionary.doc2bow(pq)

        terms = [x[0] for x in vq]
        ranking = []

        for docid, vector in enumerate(self._bag_of_words):
            vector_terms = (x[0] for x in vector)

            score = int(any(set(terms) & set(vector_terms)))

            ranking.append((docid, score))

        ranking = [x for x in sorted(ranking, key=operator.itemgetter(1), reverse=True) if x[1] > 0]
        limit = n if isinstance(n, int) else len(ranking)

        return ranking[0:limit]


class TFIndex(Index):
    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, len(self._bag_of_words))

    def create_model(self, corpus):
        self._dictionary = create_dictionary(self._corpus)
        self._bag_of_words = create_bag_of_words(self._corpus, self._dictionary, normalizer=tf_log)
        self._model = gensim.models.TfidfModel(self._bag_of_words,
                                               wlocal=tf_log,
                                               wglobal=binary_idf)
        self.index = gensim.similarities.MatrixSimilarity(self._bag_of_words,
                                                          num_features=len(self._dictionary))

    @property
    def model(self):
        return self._model

    def search(self, q, n=None, feedback=None):

        pq = pre_process_document(q)

        if feedback:
            _docs_term = {}
            _chosen_terms = []
            _counter = collections.Counter()

            for doc in self.corpus.docs:
                if doc.id in feedback:
                    _docs_term[doc.id] = set(pre_process_document(doc.body))
                    for _term in _docs_term[doc.id]:
                        _counter[_term] += 1

            for _term, _count in _counter.items():
                if _count >= len(feedback) / 2.0 and _term not in pq:
                    _chosen_terms.append(_term)
                    pq.append(_term)

            print('Using feedback terms: {}...'.format(_chosen_terms[0:10]))

        vq = self._dictionary.doc2bow(pq)
        qtfidf = self.model[vq]
        sim = self.index[qtfidf]

        ranking = [x for x in sorted(enumerate(sim), key=operator.itemgetter(1), reverse=True) if x[1] > 0]
        limit = n if isinstance(n, int) else len(ranking)

        return ranking[0:limit]


class TFIDFIndex(Index):
    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, len(self._bag_of_words))

    def create_model(self, corpus):
        self._dictionary = create_dictionary(self._corpus)
        self._bag_of_words = create_bag_of_words(self._corpus, self._dictionary)
        self._model = gensim.models.TfidfModel(self._bag_of_words)
        self.index = gensim.similarities.MatrixSimilarity(self._bag_of_words, num_features=len(self._dictionary))

    @property
    def model(self):
        return self._model

    def search(self, q, n=None, feedback=None):

        pq = pre_process_document(q)

        if feedback:
            _docs_term = {}
            _chosen_terms = []
            _counter = collections.Counter()

            for doc in self.corpus.docs:
                if doc.id in feedback:
                    _docs_term[doc.id] = set(pre_process_document(doc.body))
                    for _term in _docs_term[doc.id]:
                        _counter[_term] += 1

            for _term, _count in _counter.items():
                if _count >= len(feedback) / 2.0 and _term not in pq:
                    _chosen_terms.append(_term)
                    pq.append(_term)

            print('Using feedback terms: {}...'.format(_chosen_terms[0:10]))

        vq = self._dictionary.doc2bow(pq)
        qtfidf = self.model[vq]
        sim = self.index[qtfidf]

        ranking = [x for x in sorted(enumerate(sim), key=operator.itemgetter(1), reverse=True) if x[1] > 0]
        limit = n if isinstance(n, int) else len(ranking)

        return ranking[0:limit]


class TFIDFProbabilisticIndex(Index):
    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, len(self._bag_of_words))

    def create_model(self, corpus):
        self._dictionary = create_dictionary(self._corpus)
        self._bag_of_words = create_bag_of_words(self._corpus, self._dictionary,
                                                 normalizer=tf_log)
        self._model = gensim.models.TfidfModel(self._bag_of_words,
                                               wlocal=tf_log,
                                               wglobal=idf_probabilistic)
        self.index = gensim.similarities.MatrixSimilarity(self._bag_of_words, num_features=len(self._dictionary))

    @property
    def model(self):
        return self._model

    def search(self, q, n=None, feedback=None):

        pq = pre_process_document(q)

        if feedback:
            _docs_term = {}
            _chosen_terms = []
            _counter = collections.Counter()

            for doc in self.corpus.docs:
                if doc.id in feedback:
                    _docs_term[doc.id] = set(pre_process_document(doc.body))
                    for _term in _docs_term[doc.id]:
                        _counter[_term] += 1

            for _term, _count in _counter.items():
                if _count >= len(feedback) / 2.0 and _term not in pq:
                    _chosen_terms.append(_term)
                    pq.append(_term)

            print('Using feedback terms: {}...'.format(_chosen_terms[0:10]))

        vq = self._dictionary.doc2bow(pq)
        qtfidf = self.model[vq]
        sim = self.index[qtfidf]

        ranking = [x for x in sorted(enumerate(sim), key=operator.itemgetter(1), reverse=True) if x[1] > 0]
        limit = n if isinstance(n, int) else len(ranking)

        return ranking[0:limit]


def binary_tf(frequency):
    return int(frequency > 0)


def binary_idf(docfreq, totaldocs):
    return 1


def tf_log(tf, base=10):
    import math

    if tf == 0:
        return 0
    else:
        return 1 + math.log(tf, base)


def idf_probabilistic(df, total_docs, base=10):
    import math

    if df == 0:
        return 0
    else:
        return math.log((total_docs - df)/df, base)
