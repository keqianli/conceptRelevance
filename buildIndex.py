import sys
import logging
import re
import random
import numpy as np
from gensim.models import word2vec
import gensim
from gensim import corpora, models
from nltk.corpus import stopwords
file = '../data/data_oneFilePerLineBySection/nips/segmented_text.txt_phraseAsWord'

if len(sys.argv) > 1:
    file = sys.argv[1]

file_wordvec = file+'.wordvec'

if len(sys.argv) > 2:
    file_wordvec = sys.argv[2]

file_tfidf = file+'.tfidf'

if len(sys.argv) > 3:
    file_tfidf = sys.argv[3]


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('log')
logger.addHandler(logging.FileHandler(__file__+'.log'))
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

logger.debug('==================================')
logger.debug('for file %s' % file)

short_word = re.compile(
    r"^\w{,1}$"
)
doesnt_contain_vowel = re.compile(
    r"^[^aeiou]*$"
)

stopwordsSet = set(stopwords.words('english'))


def notMeaningfulWord(word):
    return short_word.match(word)


square_brackets_enclosed = re.compile(
    r"<phrase>(?P<phrase>[^<]*)</phrase>"
)


def trim_rule(word, count, min_count):
    if square_brackets_enclosed.match(word):
        return gensim.utils.RULE_KEEP
    if notMeaningfulWord(word):
        return gensim.utils.RULE_DISCARD
    return gensim.utils.RULE_DEFAULT
    # return gensim.utils.RULE_DEFAULT


def displayString(w):
    # return w
    return re.sub(r'</?phrase>', '', w)

valid_size = 20  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.

dictionary = {}
valid_examples_givenword = ['']


def compute_wordvec():
    for size in [200]:  # 50,128,
        for sg in [1]:  # 0
            for max_vocab_size in [None]:  # 60000,
                model_concepts_file = file_wordvec
                try:
                    model = word2vec.Word2Vec.load(model_concepts_file)
                except Exception, e:
                    print 'training new model'
                    model = word2vec.Word2Vec(word2vec.LineSentence(file), size=size,  workers=120, max_vocab_size=max_vocab_size, trim_rule=trim_rule, sg=sg)
                    model.save(model_concepts_file)

                print(model.wv.index2word[:100])

                # validation
                if dictionary == {}:
                    for _, word in enumerate(model.wv.index2word):
                        dictionary[word] = len(dictionary)
                    dictionary['UNK'] = len(dictionary)

                    valid_examples_frequent = random.sample(range(valid_window), valid_size/2)
                    valid_examples_phrase = random.sample([index for word, index in dictionary.items() if '_' in word], valid_size/2)
                    try:
                        valid_examples_frequent[0] = dictionary['analysis']
                        valid_examples_phrase[0] = dictionary['machine_learning']
                    except Exception, e:
                        pass

                    valid_examples = np.array(valid_examples_frequent + valid_examples_phrase)

                    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 10  # number of nearest neighbors
                    print('Nearest to %s: %s' % (displayString(valid_word), ', '.join([displayString(word) for word, score in model.most_similar(positive=[valid_word], topn=top_k)])))


def compute_tfidf():
    def readIntoListsOfWords(file):
        return [document.lower().split() for document in open(file).readlines()]
    wordsLists = readIntoListsOfWords(file)
    dictionary = corpora.Dictionary(wordsLists)
    corpus = [dictionary.doc2bow(text) for text in wordsLists]
    modelTfidf = models.TfidfModel(corpus)

    corpora.MmCorpus.serialize(file_tfidf + '.corpus', corpus)  # store to disk, for later use
    dictionary.save(file_tfidf + '.dict')  # store the dictionary, for future reference
    modelTfidf.save(file_tfidf+'.modelTfidf')


def main():
    compute_wordvec()


if __name__ == '__main__':
    main()
