import json
from functools import partial
from collections import defaultdict

import numpy as np

from clf.indexer import Indexer
from clf.model import TypeClassifier
from clf.prediction import find_type, get_type_scores, predict_all_words

WORDS_FILE_PATH = "/Users/konix/embeddings_dep/deps.300d.words"  # "/Users/konix/Workspace/nerlstm/glove/glove.100d.words"
VECTORS_FILE_PATH = "/Users/konix/embeddings_dep/deps.300d.vectors"  # "/Users/konix/Workspace/nerlstm/glove/glove.100d.vectors"
EMBEDDINGS_FILE_PATH = "/Users/konix/embeddings_dep/deps.words"  # "/Users/konix/Workspace/nerlstm/glove/glove.6B.100d.txt"
WORD_TO_TYPE_LIST_FILE_PATH = "/Users/konix/Workspace/embedding_clf/resources/word_to_type_name_list.json"


RELEVANT_TYPES = {
    'person',
    'artifact',
    'act',
    'location',
    'animal',
    'state',
    'plant',
    'substance',
    'food'
}

INCLUDE_TYPE_OTHER = True


def eavl_clf(clf, word_indexer, type_indexer, word_to_type):
    type_to_result = defaultdict(lambda: {'true_positive': 0.0, 'false_positive': 0.0, 'count': 0.0})
    transition_matrix = defaultdict(lambda: defaultdict(int))

    for word_text, word_type_list in word_to_type.iteritems():
        word_type = word_type_list[0]
        type_to_result[word_type]['count'] += 1

        clf_type = find_type(clf, word_indexer, type_indexer, word_text)
        clf_result = 'true_positive' if word_type == clf_type else 'false_positive'
        type_to_result[clf_type][clf_result] += 1

        transition_matrix[word_type][clf_type] += 1

    type_to_result = dict(type_to_result)
    for type, type_result in type_to_result.iteritems():
        clf_count = (type_result['true_positive'] + type_result['false_positive'])
        accuracy = type_result['true_positive'] / clf_count if clf_count > 0 else None
        recall = (type_result['true_positive'] / type_result['count']) if type_result['count'] > 0 else None
        f_measure = (2*accuracy*recall) / (accuracy + recall) if (accuracy and recall) else None

        type_result['accuracy'] = accuracy
        type_result['recall'] = recall
        type_result['f_measure'] = f_measure
    return type_to_result, transition_matrix


def main():
    external_word_embeddings = {}
    for line in open(EMBEDDINGS_FILE_PATH, 'rb').readlines():
        word, embedding_str = line.rstrip().split(' ', 1)
        word = word.lower()
        embedding = np.asarray([float(value_str) for value_str in embedding_str.split()])
        external_word_embeddings[word] = embedding

    with open(WORD_TO_TYPE_LIST_FILE_PATH, 'rb') as word_to_type_list_file_obj:
        word_to_type_list = json.load(word_to_type_list_file_obj)

    lower_word_to_type_set = {}
    for word, word_type_list in word_to_type_list.iteritems():
        lower_word = word.lower()

        word_type_set = lower_word_to_type_set.get(lower_word, set())
        word_type_set.update(word_type_list)

        word_type_set = {word_type_ for word_type_ in word_type_set if word_type_ in RELEVANT_TYPES}
        if not word_type_set:
            if INCLUDE_TYPE_OTHER:
                word_type_set = {'other'}
            else:
                continue
        lower_word_to_type_set[lower_word] = word_type_set

    common_words = set(lower_word_to_type_set.keys()).intersection(set(external_word_embeddings.keys()))
    print "Found %d common words" % len(common_words)

    common_words_with_single_type = []
    for word, word_type_set in lower_word_to_type_set.iteritems():
        if word in common_words and len(word_type_set) == 1:
            common_words_with_single_type.append(word)
    print "Found %d common words with single type" % len(common_words_with_single_type)

    word_to_type = {}
    type_to_word_list = {}
    for word in common_words_with_single_type:
        word_type = next(iter(lower_word_to_type_set[word]))
        if word_type not in type_to_word_list:
            type_to_word_list[word_type] = []

        word_to_type[word] = [word_type]
        type_to_word_list[word_type].append(word)

    type_indexer = Indexer()
    type_indexer.index_object_list(type_to_word_list.keys())

    word_indexer = Indexer()
    word_indexer.index_object_list(external_word_embeddings.keys())

    cutoff = int(0.8 * len(word_to_type))
    word_to_type_items = word_to_type.items()
    word_to_type_train = dict(word_to_type_items[:cutoff])
    word_to_type_test = dict(word_to_type_items[cutoff:])
    print "Train size: %d, Test size: %d" % (len(word_to_type_train), len(word_to_type_test))

    my_clf = TypeClassifier(word_indexer, type_indexer, external_word_embeddings)
    my_clf.train(word_to_type_train, iterations=250)

    W = np.array(file(WORDS_FILE_PATH).read().strip().split())
    w2i = {w_: i_ for (i_, w_) in enumerate(W)}
    E = np.loadtxt(VECTORS_FILE_PATH)
    prediction_matrix = predict_all_words(my_clf, E)

    data_set_to_result = {
        "train": eavl_clf(my_clf, word_indexer, type_indexer, word_to_type_train),
        "test": eavl_clf(my_clf, word_indexer, type_indexer, word_to_type_test)
    }
    for data_set_name, data_set_result in data_set_to_result.iteritems():
        print data_set_name.upper()

        for type, type_result in data_set_result[0].iteritems():

            accuracy = type_result['accuracy']
            recall = type_result['recall']
            f_measure = type_result['f_measure']

            acc_str = "acc=%.2f" % accuracy if accuracy else "acc=N/A "
            recall_str = "recall=%.2f" % recall if recall else "recall=N/A "
            f1_str = "f1=%.2f" % f_measure if f_measure else "f1=N/A "
            print "%10s: %s, %s, %s" % (type, acc_str, recall_str, f1_str)

        print "------------------\n"

    import IPython;IPython.embed()

    # word_by_sentiment = {'positive': [], 'negative': []}
    # positive_idx, negative_idx = type_indexer.get_index('positive'), type_indexer.get_index('negative')
    # for w, i in w2i.iteritems():
    #     type_scores = prediction_matrix[i]
    #     pos_score, neg_score = type_scores[positive_idx], type_scores[negative_idx]
    #
    #     if neg_score >= 0.5 and (pos_score < 0.1 or ((neg_score - pos_score) > 0.5)):
    #         word_by_sentiment['negative'].append(w)
    #     elif pos_score >= 0.5 and (neg_score < 0.1 or ((pos_score - neg_score) > 0.5)):
    #         word_by_sentiment['positive'].append(w)
    #
    # with open(CLASSIFICATION_FILE_PATH, 'wb') as classification_file:
    #     json.dump(word_by_sentiment, classification_file)
    #
    # my_find_type = partial(find_type, my_clf, word_indexer, type_indexer)
    # my_get_type_scores = partial(get_type_scores, my_clf, word_indexer, type_indexer)
    # import IPython;IPython.embed()


if __name__ == '__main__':
    main()
