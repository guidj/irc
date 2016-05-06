import math
import collections

import irc.models


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

                    document = irc.models.Document(document_id, '\n'.join(sentences))
                    documents.append(document)

                document_id = int(line.replace(id_mark, '').strip())
            elif line.startswith(body_mark):
                sentences = []
            else:
                sentences.append(line)

    if document_id:
        document = irc.models.Document(document_id, '\n'.join(sentences))
        documents.append(document)

    corpus = irc.models.Corpus(docs=documents)

    return corpus


def read_queries(filepath):

    id_mark = '.I'
    body_mark = '.W'

    queries = []

    with open(filepath, 'r') as fp:

        sentences = []
        query_id = None

        for line in fp:

            if line.startswith(id_mark):

                # save previous query
                if query_id:
                    query = irc.models.Query(query_id, '\n'.join(sentences).strip())
                    queries.append(query)

                query_id = int(line.replace(id_mark, '').strip())
            elif line.startswith(body_mark):
                sentences = []
            else:
                sentences.append(line)

        if query_id:
            query = irc.models.Query(query_id, '\n'.join(sentences).strip())
            queries.append(query)

    return queries


def read_relevance(filepath):

    relevance = collections.defaultdict(set)

    with open(filepath, 'r') as fp:

        for line in fp:
            tokens = line.split(' ')
            _id, _doc = int(tokens[0]), int(tokens[2])

            relevance[_id].add(_doc)

    return relevance


def binary_tf(frequency):

    return int(frequency > 0)


def binary_idf(docfreq, totaldocs):

    return 1


def tf_log(tf, base=10):

    if tf == 0:
        return 0
    else:
        return 1 + math.log(tf, base)


def idf_smooth(idf, base=10):

    if idf == 0:
        return 0
    else:
        return 1 + math.log(1 + idf, base)
