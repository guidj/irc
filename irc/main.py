import os.path

from . import utils
from . import models
from . import contants


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

        if inp[i] in ('--corpora', '-corpora'):
            if i + 1 >= argc:
                raise RuntimeError('Missing value for parameter --corpora')

            if not os.path.isfile(inp[i+1]):
                raise RuntimeError('{} is not a valid file path'.format(inp[i+1]))

            args['corpora'] = inp[i+1]

        elif inp[i] in ('--index', '-index'):
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

    required_params = ('corpora', 'index', 'q')

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


if __name__ == '__main__':

    import os
    import sys

    try:
        args = parse_args(sys.argv[1:])
        corpus = utils.read_corpus_from_file(args['corpora'])

        index = model(args['index'])(corpus)

        index.search(q=args['q'], n=args['n'])

    except RuntimeError as err:
        print err
        print usage()
        exit(1)
