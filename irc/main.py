import os.path

from . import utils
from . import models
from . import contants
from . import config


def usage():

    msg = """
    Usage:
        python2.7 -m irc.main --corpora [MED.ALL] --index [index] --q [term] --n [10]

    Where:

        --corpora/-corpora  : Path to the MED.ALL file
        --index             : Index type. Options: Binary, TF, TF-IDF, TF-IDF-S
        --q                 : Search query term
        --n                 : Number of matches to be returned. Default is 10, * for all
    """

    print msg


def parse_args(inp):

    args = {}
    argc = len(inp)

    for i in range(0, argc):

        if inp[i] in ('--index', '-index'):
            if i + 1 >= argc:
                raise RuntimeError('Missing value for parameter --index')

            if inp[i+1] not in contants.INDEX_MODELS:
                raise RuntimeError('`{}` is not a valid Index model'.format(inp[i+1]))

            args['index'] = inp[i+1]

        elif inp[i] in ('--q', '-q'):
            if i + 1 >= argc:
                raise RuntimeError('Missing value for parameter --q')

            args['q'] = inp[i + 1]

        elif inp[i] in ('--n', '-n'):
            if i + 1 >= argc:
                raise RuntimeError('Missing value for parameter --n')

            try:
                args['n'] = int(inp[i+1])
            except ValueError:
                if inp[i+1] == "all":
                    args['n'] = None
                else:
                    raise RuntimeError('Unknown value for parameter -n')

    required_params = ('index', 'q')

    for param in required_params:
        if param not in args:
            raise RuntimeError('Missing parameter {}'.format(param))

    if 'n' not in args:
        args['n'] = 10

    return args


def model(name):

    return {
        'Binary': models.BinaryIndex,
        'TF': models.TFIndex,
        'TF-IDF': models.TFIDFIndex,
        'TF-IDF-S': models.TFIDFSmoothIndex
    }[name]


def pprint(index, ranking):
    print index
    for i, rank in enumerate(ranking):
        print "[Rank = {:3}, Score = {:.3f}] {}".format(i + 1, rank[1], index.corpus.docs[rank[0]])


if __name__ == '__main__':

    import os
    import sys

    try:
        args = parse_args(sys.argv[1:])
        corpuspath = os.path.join([config.CORPUS_DIR, contants.MED_CORPUS])

        corpus = utils.read_corpus_from_file(args['corpora'])

        index = model(args['index'])(corpus)

        results = index.search(q=args['q'], n=args['n'])

        pprint(index, ranking=results)

    except RuntimeError as err:
        print err
        print usage()
        exit(1)
