import random

import numpy as np
import dynet as dy


class TypeClassifier(object):
    # WORD_DIM = 100
    # HIDDEN_DIM = 25
    WORD_DIM = 300
    HIDDEN_DIM = 50

    def __init__(self, word_indexer, type_indexer, external_word_embeddings=None):
        self.word_indexer = word_indexer
        self.type_indexer = type_indexer
        self.external_word_embeddings = external_word_embeddings

        model = dy.Model()
        self.word_lookup = model.add_lookup_parameters((len(word_indexer), self.WORD_DIM))

        if external_word_embeddings:
            for idx in xrange(len(word_indexer)):
                word = word_indexer.get_object(idx)
                if word in external_word_embeddings:
                    self.word_lookup.init_row(idx, external_word_embeddings[word])

        self.param_hidden = model.add_parameters((self.HIDDEN_DIM, self.WORD_DIM))
        self.param_out = model.add_parameters((len(type_indexer), self.HIDDEN_DIM))

        self.model = model
        self.trainer = dy.AdamTrainer(model)

    def train(self, train_word_to_type, test_word_to_type=None, iterations=50):
        training_examples = self.build_example_vectors(train_word_to_type)

        for iteration_idx in xrange(1, iterations+1):
            print "Starting training iteration %d/%d" % (iteration_idx, iterations)
            random.shuffle(training_examples)
            loss = 0

            for example_index, (word_index, expected_output) in enumerate(training_examples, 1):
                out_expression = self.build_expression(word_index)

                expected_output_expr = dy.vecInput(len(self.type_indexer))
                expected_output_expr.set(expected_output)
                sentence_error = dy.squared_distance(out_expression, expected_output_expr)
                # sentence_error = dy.pickneglogsoftmax(out_expression, np.argmax(expected_output))

                loss += sentence_error.scalar_value()
                sentence_error.backward()
                self.trainer.update()

            # Trainer Status
            self.trainer.status()
            print loss / float(len(training_examples))

    def build_example_vectors(self, word_to_type):
        examples = []
        for word, word_gazetteers in word_to_type.iteritems():
            word_index = self.word_indexer.get_index(word)
            word_type_indices = [1 if self.type_indexer.get_object(type_idx) in word_gazetteers else 0
                                 for type_idx in xrange(len(self.type_indexer))]
            examples.append((word_index, np.asarray(word_type_indices)))
        return examples

    def build_expression(self, word_index):
        dy.renew_cg()
        word_vector = dy.lookup(self.word_lookup, word_index, update=False)
        return self.param_out * dy.tanh(self.param_hidden * word_vector)
