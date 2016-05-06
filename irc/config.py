import os.path

import irc.contants

PROJECT_BASE = ''.join([os.path.dirname(os.path.abspath(__file__)), "/../"])
CORPUS_DIR = os.path.join(PROJECT_BASE, 'corpus')
FILES = {
    'corpus': os.path.join(CORPUS_DIR, irc.contants.MED_CORPUS),
    'queries': os.path.join(CORPUS_DIR, irc.contants.MED_QUERY),
    'relevance': os.path.join(CORPUS_DIR, irc.contants.MED_RELEVANCE),
}