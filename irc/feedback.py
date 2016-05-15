from irc import config
from irc import constants
from irc import evaluation
from irc import utils
from irc import domain


def usage():
    msg = """
    Usage:  python -m irc.feedback --index [index] --q [ID]

    Where:

        --index             : Index type. Options: Binary, TF, TF-IDF, TF-IDF-Prob
        --q                 : Query: 1-30
        --n                 : Number of matches to be returned. Default is 10, * for all
    """

    print(msg)


def parse_args(inp):
    args = {}
    argc = len(inp)

    for i in range(0, argc):

        if inp[i] in ('--index', '-index'):
            if i + 1 >= argc:
                raise RuntimeError('Missing value for parameter --index')

            if inp[i + 1] not in constants.INDEX_MODELS:
                raise RuntimeError('`{}` is not a valid Index model'.format(inp[i + 1]))

            args['index'] = inp[i + 1]

        elif inp[i] in ('--q', '-q'):
            if i + 1 >= argc:
                raise RuntimeError('Missing value for parameter --q')

            try:
                args['q'] = int(inp[i + 1])
            except ValueError:
                raise RuntimeError(
                    'Invalid --q value: {}. Should be integer between 1 and 30, inclusive'.format(inp[i + 1])
                )

    required_params = ('index', 'q')

    for param in required_params:
        if param not in args:
            raise RuntimeError('Missing parameter {}'.format(param))

    if 'n' not in args:
        args['n'] = 15

    return args


def pprint(index, ranking):
    print(index)
    for i, rank in enumerate(ranking):
        print("[Rank = {:3}, Score = {:.3f}] {}".format(i + 1, rank[1], index.corpus.docs[rank[0]]))


def evaluate_query(query, ranking, relevance):

    assert isinstance(query, domain.Query)

    query.num_docs_retrieved = len(ranking)
    query.relevant_docs = relevance[query.id]

    num_relevant_docs_retrieved = 0
    std_precisions, std_recalls = [], [x / float(10) for x in range(0, 10 + 1, 1)]
    precisions, recalls = [], []

    for i, rank in enumerate(ranking):

        _doc_id, _doc_score = rank

        if _doc_id in query.relevant_docs:
            num_relevant_docs_retrieved += 1

        _precision = float(num_relevant_docs_retrieved) / (i + 1)
        _recall = float(num_relevant_docs_retrieved) / len(query.relevant_docs)
        precisions.append(_precision)
        recalls.append(_recall)

    query.num_relevant_docs_retrieved = num_relevant_docs_retrieved
    query.precision = float(num_relevant_docs_retrieved) / len(ranking)
    query.recall = float(num_relevant_docs_retrieved) / len(query.relevant_docs)

    for i, std_recall in enumerate(std_recalls):
        std_precision = max([0] + [x for j, x in enumerate(precisions) if recalls[j] >= std_recall])
        std_precisions.append(std_precision)

    return std_precisions, std_recalls


def mkfigure(scores):
    import os.path

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    plt.clf()

    for _v in scores:
        plt.plot(_v[1][1], _v[1][0], '--', linewidth=2, label=str(_v[0]))

    plt.title('Feedback System Evaluation')
    plt.xlabel('Recall')
    plt.ylabel('Precision at recall level x')
    plt.legend()

    figname = os.path.join(config.PROJECT_BASE, 'img/feedback.png')

    if os.path.exists(figname):
        os.remove(figname)

    if not os.path.exists(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    plt.savefig(figname)

    plt.close()

    print('Updated feedback curve @{}'.format(figname))


if __name__ == '__main__':

    import sys

    try:
        args = parse_args(sys.argv[1:])
        scores = []
        turn = 0

        print('Loading corpus...')

        corpus = utils.read_corpus_from_file(config.FILES['corpus'])
        queries = {query.id: query for query in utils.read_queries(config.FILES['queries'])}
        relevance = utils.read_relevance(config.FILES['relevance'])

        index = evaluation.model(args['index'])(corpus)
        q = queries[args['q']]
        results = index.search(q=q.term, n=args['n'])

        scores.append((turn, evaluate_query(q, results, relevance)))
        turn += 1

        pprint(index, ranking=results)
        print('Precision({:.3f}), Recall({:.3f})'.format(q.precision, q.recall))

        mkfigure(scores)

        while True:

            try:
                inp = input('\nChoose top N results to give feedback: ')
                n = int(inp)
            except ValueError:
                print('[ERROR] N should an integer between 1 and {}. Try again...'.format(len(results)))

            else:

                focused = [rank[0] for rank in results[0:n]]
                pprint(index, results[0:n])

                while True:
                    try:
                        inp = input('\nList the IDs of the relevant docs (e.g. 1, 13, 46): ')
                        relevant = [int(x) for x in inp]
                    except ValueError:
                        print('[ERROR]: The IDs should an integers. Try again...')
                        continue
                    else:

                        feedback = set(focused) & set(relevant)
                        print(
                            '[WARN] Ignoring docs not present in top {}: {}'.format(
                                n, [x for x in feedback if x not in relevant]
                            )
                        )

                        results = index.search(q=q.term, feedback=feedback)
                        pprint(index, ranking=results)

                        scores.append((turn, evaluate_query(q, results, relevance)))
                        print('Precision({:.3f}), Recall({:.3f})'.format(q.precision, q.recall))
                        mkfigure(scores)
                        turn += 1
                        break

                        # TODO: precision & recall

    except RuntimeError as err:
        print(err)
        usage()
        exit(1)
