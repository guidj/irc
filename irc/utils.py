import collections
import pickle
import os.path


def save_object(obj, filename):
    from irc import config

    path = os.path.join(config.DATA_DIR, filename)
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp, pickle.HIGHEST_PROTOCOL)


def retrieve_object(filename):
    from irc import config

    path = os.path.join(config.DATA_DIR, filename)

    try:
        with open(path, 'rb') as fp:
            d = pickle.load(fp)
    except IOError:
        return None
    else:
        return d


def read_corpus_from_file(filepath):
    from irc import domain

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

                    document = domain.Document(document_id, '\n'.join(sentences))
                    documents.append(document)

                document_id = int(line.replace(id_mark, '').strip())
            elif line.startswith(body_mark):
                sentences = []
            else:
                sentences.append(line)

    if document_id:
        document = domain.Document(document_id, '\n'.join(sentences))
        documents.append(document)

    corpus = domain.Corpus(docs=documents)

    return corpus


def read_queries(filepath):
    from irc import domain

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
                    query = domain.Query(query_id, '\n'.join(sentences).strip())
                    queries.append(query)

                query_id = int(line.replace(id_mark, '').strip())
            elif line.startswith(body_mark):
                sentences = []
            else:
                sentences.append(line)

        if query_id:
            query = domain.Query(query_id, '\n'.join(sentences).strip())
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

