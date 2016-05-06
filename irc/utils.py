import models
import math


def read_corpus_from_file(filepath):

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

                    document = models.Document(document_id, '\n'.join(sentences))
                    documents.append(document)

                document_id = int(line.replace(id_mark, '').strip())
            elif line.startswith(body_mark):
                sentences = []
            else:
                sentences.append(line)

    corpus = models.Corpus(docs=documents)

    return corpus


def binary_tf(frequency):

    return int(frequency > 0)


def binary_idf(docfreq, totaldocs):

    return 1


def tf_log(tf, base=10):

    if tf == 0:
        return 0
    else:
        return 1 + math.log(tf, base)


def idf_smooth(idf, base=10):

    if idf == 0:
        return 0
    else:
        return 1 + math.log(1 + idf, base)
