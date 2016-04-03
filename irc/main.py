import utils
import models


if __name__ == '__main__':

    import os
    import sys

    filepath = sys.argv[1]
    search_term = sys.argv[2]

    corpus = utils.read_corpus_from_file(filepath)

    tfidf_smooth_index = models.TFIDFSmoothIndex(corpus)
    tfidf_index = models.TFIDFIndex(corpus)
    tf_index = models.TFIndex(corpus)
    binary_index = models.BinaryIndex(corpus)

    tfidf_smooth_index.search(search_term)

    print '\n'

    tfidf_index.search(search_term)

    print '\n'

    tf_index.search(search_term)

    print '\n'

    binary_index.search(search_term)






