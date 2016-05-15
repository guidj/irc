from irc import constants
from irc import config
from irc import utils
from irc import evaluation


def usage():

    msg = """
    Usage:  python2.7 -m irc.feedback --index [index] --q [ID]

    Where:

        --index             : Index type. Options: Binary, TF, TF-IDF, TF-IDF-S
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

            if inp[i+1] not in constants.INDEX_MODELS:
                raise RuntimeError('`{}` is not a valid Index model'.format(inp[i+1]))

            args['index'] = inp[i+1]

        elif inp[i] in ('--q', '-q'):
            if i + 1 >= argc:
                raise RuntimeError('Missing value for parameter --q')

            try:
                args['q'] = int(inp[i + 1])
            except ValueError:
                raise RuntimeError(
                    'Invalid --q value: {}. Should be integer between 1 and 30, inclusive'.format(inp[i+1])
                )

    required_params = ('index', 'q')

    for param in required_params:
        if param not in args:
            raise RuntimeError('Missing parameter {}'.format(param))

    if 'n' not in args:
        args['n'] = 10

    return args


def pprint(index, ranking):
    print(index)
    for i, rank in enumerate(ranking):
        print("[Rank = {:3}, Score = {:.3f}] {}".format(i + 1, rank[1], index.corpus.docs[rank[0]]))


if __name__ == '__main__':

    import sys

    try:
        args = parse_args(sys.argv[1:])

        print('Loading corpus...')

        corpus = utils.read_corpus_from_file(config.FILES['corpus'])
        queries = {query.id: query for query in utils.read_queries(config.FILES['queries'])}

        index = evaluation.model(args['index'])(corpus)
        q = queries[args['q']]
        results = index.search(q=q.term)

        pprint(index, ranking=results)

        while True:

            try:
                inp = input('Choose top N results to give feedback: ')
                n = int(inp)
            except ValueError:
                print('[ERROR] N should an integer between 1 and {}. Try again...'.format(len(results)))

            else:

                focused = [rank[0] for rank in results[0:n]]
                pprint(index, results[0:n])

                while True:
                    try:
                        inp = input('List the IDs of the relevant docs (e.g. 1, 13, 46): ')
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
                        break

                # TODO: precision & recall

    except RuntimeError as err:
        print(err)
        usage()
        exit(1)
