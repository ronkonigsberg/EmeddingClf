import numpy as np


def find_type(my_clf, word_indexer, type_indexer, word_text):
    word_index = word_indexer.get_index(word_text)
    type_vector = my_clf.build_expression(word_index).npvalue()
    min_index, min_diff = None, None
    for type_index, type_value in enumerate(type_vector):
        cur_diff = abs(type_value - 1)
        if min_diff is None or cur_diff < min_diff:
            min_diff = cur_diff
            min_index = type_index
    return type_indexer.get_object(min_index)


def get_type_scores(my_clf, word_indexer, type_indexer, word_text):
    word_index = word_indexer.get_index(word_text)
    type_vector = my_clf.build_expression(word_index).npvalue()
    type_to_score = {}
    for type_index, type_value in enumerate(type_vector):
        type_name = type_indexer.get_object(type_index)
        type_to_score[type_name] = type_value
    return type_to_score


def predict_all_words(my_clf, E):
    H = np.tanh(np.dot(E, np.transpose(my_clf.param_hidden.as_array())))
    return np.dot(H, np.transpose(my_clf.param_out.as_array()))
