import os.path

import pandas

import irc.contants
import irc.config
import irc.utils
import irc.models


def evaluate_index(index):

    assert isinstance(index, irc.models.Index)

    queries = irc.utils.read_queries(irc.config.FILES['queries'])
    relevance = irc.utils.read_relevance(irc.config.FILES['relevance'])

    for query in queries:
        assert isinstance(query, irc.models.Query)

        ranking = index.search(query.term, n=None)
        query.num_docs_retrieved = len(ranking)
        query.relevant_docs = relevance[query.id]

        num_relevant_docs_retrieved = 0
        std_precisions, std_recalls = [], [x/float(10) for x in range(0, 10+1, 1)]
        precisions, recalls = [], []

        for i, rank in enumerate(ranking):

            _doc_id, _doc_score = rank

            if _doc_id in query.relevant_docs:
                num_relevant_docs_retrieved += 1

            _precision = float(num_relevant_docs_retrieved)/(i+1)
            _recall = float(num_relevant_docs_retrieved)/len(query.relevant_docs)
            precisions.append(_precision)
            recalls.append(_recall)

        query.num_relevant_docs_retrieved = num_relevant_docs_retrieved
        query.precision = float(num_relevant_docs_retrieved)/len(ranking)
        query.recall = float(num_relevant_docs_retrieved)/len(query.relevant_docs)

        for i, std_recall in enumerate(std_recalls):
            std_precision = max([0] + [x for j, x in enumerate(precisions) if recalls[j] >= std_recall])
            query.evaluation.loc[i] = [std_precision, std_recall]

    return queries


def model(name):

    return {
        'Binary': irc.models.BinaryIndex,
        'TF': irc.models.TFIndex,
        'TF-IDF': irc.models.TFIDFIndex,
        'TF-IDF-S': irc.models.TFIDFSmoothIndex
    }[name]


if __name__ == '__main__':

    import sys

    corpus = irc.utils.read_corpus_from_file(irc.config.FILES['corpus'])

    m = model(sys.argv[1])(corpus)

    evaluate_index(m)
