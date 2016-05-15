import os.path

import irc.constants

PROJECT_BASE = ''.join([os.path.dirname(os.path.abspath(__file__)), "/../"])
CORPUS_DIR = os.path.join(PROJECT_BASE, 'corpus')
DATA_DIR = os.path.join(PROJECT_BASE, 'db')
FILES = {
    'corpus': os.path.join(CORPUS_DIR, irc.constants.MED_CORPUS),
    'queries': os.path.join(CORPUS_DIR, irc.constants.MED_QUERY),
    'relevance': os.path.join(CORPUS_DIR, irc.constants.MED_RELEVANCE),
}
