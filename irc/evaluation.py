import os
import os.path

from irc import config
from irc import utils
from irc import domain


def usage():

    msg = """
    Usage:  python -m irc.evaluation
    """

    print(msg)


def pprint(index, ranking):
    print(index)
    for i, rank in enumerate(ranking):
        print("[Rank = {:3}, Score = {:.3f}] {}".format(i + 1, rank[1], index.corpus.docs[rank[0]]))


def evaluate_index(index):
    print(index)
    assert isinstance(index, domain.Index)

    queries = utils.read_queries(config.FILES['queries'])
    relevance = utils.read_relevance(config.FILES['relevance'])

    for query in queries:
        assert isinstance(query, domain.Query)

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


def summary(queries):

    n = 11
    m = {
        'precision': [0 for _ in range(0, n)],
        'recall': [0 for _ in range(0, n)]
    }

    for query in queries:
        for i in range(0, n):
            m['precision'][i] += query.evaluation.loc[i]['precision']
            m['recall'][i] = query.evaluation.loc[i]['recall']

    for i in range(0, n):
        m['precision'][i] /= n
        m['recall'][i] /= n

    return m


def model(name):

    return {
        'Binary': domain.BinaryIndex,
        'TF': domain.TFIndex,
        'TF-IDF': domain.TFIDFIndex,
        'TF-IDF-Prob': domain.TFIDFProbabilisticIndex
    }[name]


def mkfigure(metrics):

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    for _name, _data in metrics.items():
        plt.plot(_data['recall'], _data['precision'], '--', linewidth=2, label=_name)

    plt.title('IR Evaluation')
    plt.xlabel('Recall')
    plt.ylabel('Precision at recall level x')
    plt.legend()

    figname = os.path.join(config.PROJECT_BASE, 'img/evaluation.png')

    if not os.path.exists(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    plt.savefig(figname)
    plt.close()

    print('Image saved @{}'.format(figname))


if __name__ == '__main__':

    print('Loading corpus...')

    corpus = utils.read_corpus_from_file(config.FILES['corpus'])

    available_models = {
        'TF': domain.TFIndex,
        'TF-IDF': domain.TFIDFIndex,
        'TF-IDF-Prob': domain.TFIDFProbabilisticIndex
    }

    index_models = {}

    print('Building models...')

    for k, model_class in available_models.items():
        index_models[k] = model_class(corpus)

    print('Running evaluations...')

    results = map(lambda x, y: (x, evaluate_index(y)), index_models.keys(), (v for _, v in index_models.items()))

    metrics = {}

    for name, queries in results:
        metrics[name] = summary(queries)

    mkfigure(metrics)
