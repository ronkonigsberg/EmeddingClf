import json
from functools import partial

import numpy as np

from clf.indexer import Indexer
from clf.model import TypeClassifier
from clf.prediction import find_type, get_type_scores, predict_all_words
from clf.consts import (EMBEDDINGS_FILE_PATH, WORDS_FILE_PATH, VECTORS_FILE_PATH, COMMON_WORDS_FILE_PATH,
                        CLASSIFICATION_FILE_PATH, WORD_TYPE_TO_LIST, NLTK_STOP_WORDS)


def main():
    external_word_embeddings = {}
    for line in open(EMBEDDINGS_FILE_PATH, 'rb').readlines():
        word, embedding_str = line.rstrip().split(' ', 1)
        word = word.lower()
        embedding = np.asarray([float(value_str) for value_str in embedding_str.split()])
        external_word_embeddings[word] = embedding

    type_to_word_list = {}
    for word_type, word_list_file_path in WORD_TYPE_TO_LIST.iteritems():
        word_list = open(word_list_file_path, 'rb').read().split('\n')[:-1]
        word_list = [word_text_ for word_text_ in word_list if word_text_ in external_word_embeddings]
        type_to_word_list[word_type] = word_list

    word_to_type = {}
    for word_type, word_list in type_to_word_list.iteritems():
        for word_text in word_list:
            word_to_type[word_text] = [word_type]

    for word_text in NLTK_STOP_WORDS:
        if word_text in external_word_embeddings:
            word_to_type[word_text] = []

    common_words = open(COMMON_WORDS_FILE_PATH, 'rb').read().split('\n')
    for word_text in common_words:
        if word_text in external_word_embeddings and word_text not in word_to_type:
            word_to_type[word_text] = []

    type_indexer = Indexer()
    type_indexer.index_object_list(WORD_TYPE_TO_LIST.keys())

    word_indexer = Indexer()
    word_indexer.index_object_list(external_word_embeddings.keys())

    my_clf = TypeClassifier(word_indexer, type_indexer, external_word_embeddings)
    my_clf.train(word_to_type, iterations=1000)

    W = np.array(file(WORDS_FILE_PATH).read().strip().split())
    w2i = {w_: i_ for (i_, w_) in enumerate(W)}
    E = np.loadtxt(VECTORS_FILE_PATH)

    prediction_matrix = predict_all_words(my_clf, E)

    word_by_sentiment = {'positive': [], 'negative': []}
    positive_idx, negative_idx = type_indexer.get_index('positive'), type_indexer.get_index('negative')
    for w, i in w2i.iteritems():
        type_scores = prediction_matrix[i]
        pos_score, neg_score = type_scores[positive_idx], type_scores[negative_idx]

        if neg_score >= 0.5 and (pos_score < 0.1 or ((neg_score - pos_score) > 0.5)):
            word_by_sentiment['negative'].append(w)
        elif pos_score >= 0.5 and (neg_score < 0.1 or ((pos_score - neg_score) > 0.5)):
            word_by_sentiment['positive'].append(w)

    with open(CLASSIFICATION_FILE_PATH, 'wb') as classification_file:
        json.dump(word_by_sentiment, classification_file)

    my_find_type = partial(find_type, my_clf, word_indexer, type_indexer)
    my_get_type_scores = partial(get_type_scores, my_clf, word_indexer, type_indexer)
    import IPython;IPython.embed()


if __name__ == '__main__':
    main()
