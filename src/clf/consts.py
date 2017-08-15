from os import path


BASE_DIR = path.dirname(path.dirname(path.dirname(__file__)))
RESOURCES_DIR = path.join(BASE_DIR, 'resources')


# EMBEDDINGS_FILE_PATH = '/Users/konix/Workspace/GloVe-1.2/yelp_vectors.txt'
# WORDS_FILE_PATH = "/Users/konix/Workspace/GloVe-1.2/yelp.50d.words"
# VECTORS_FILE_PATH = "/Users/konix/Workspace/GloVe-1.2/yelp.50d.vectors"
# COMMON_WORDS_FILE_PATH = "/Users/konix/Workspace/nerlstm/glove/yelp_common"
# CLASSIFICATION_FILE_PATH = "/Users/konix/Workspace/nerlstm/glove/yelp_classification_2017_07_01_test2"

# EMBEDDINGS_FILE_PATH = '/Users/konix/Workspace/GloVe-1.2/amazon_vectors.txt'
# WORDS_FILE_PATH = "/Users/konix/Workspace/GloVe-1.2/amazon.50d.words"
# VECTORS_FILE_PATH = "/Users/konix/Workspace/GloVe-1.2/amazon.50d.vectors"
# COMMON_WORDS_FILE_PATH = "/Users/konix/Workspace/nerlstm/glove/amazon_common"
# CLASSIFICATION_FILE_PATH = "/Users/konix/Workspace/nerlstm/glove/amazon_classification_2017_07_01"

# EMBEDDINGS_FILE_PATH = '/Users/konix/Workspace/nerlstm/glove/glove.6B.100d.txt'
# WORDS_FILE_PATH = "/Users/konix/Workspace/nerlstm/glove/glove.100d.words"
# VECTORS_FILE_PATH = "/Users/konix/Workspace/nerlstm/glove/glove.100d.vectors"
# COMMON_WORDS_FILE_PATH = None
# CLASSIFICATION_FILE_PATH = "/Users/konix/Workspace/nerlstm/glove/glove_embd"


WORDS_FILE_PATH = "/Users/konix/embeddings_dep/deps.300d.words"
VECTORS_FILE_PATH = "/Users/konix/embeddings_dep/deps.300d.vectors"
EMBEDDINGS_FILE_PATH = "/Users/konix/embeddings_dep/deps.words"
COMMON_WORDS_FILE_PATH = None
CLASSIFICATION_FILE_PATH = "/Users/konix/Workspace/nerlstm/glove/dep_embd"


WORD_TYPE_TO_LIST = {
    'positive': path.join(RESOURCES_DIR, 'positive.txt'),
    'negative': path.join(RESOURCES_DIR, 'negative.txt'),
}


NLTK_STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                   'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                   'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                   'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                   'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                   'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                   'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                   'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                   'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                   'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                   'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                   'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
