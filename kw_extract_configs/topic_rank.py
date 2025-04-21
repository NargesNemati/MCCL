import re
import nltk
import logging
from itertools import product
import networkx as nx
from nltk.stem import SnowballStemmer
from collections import defaultdict
from langdetect import detect
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.hierarchy import linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

"""set of parts of speech that constitutes a keyphrase
  Anette Hulth. 2003. Improved Automatic Keyword
  Extraction Given More Linguistic Knowledge. In
  Proceedings of the 2003 Conference on Empirical
  Methods in Natural Language Processing, pages
  216–223, Stroudsburg, PA, USA. Association for
  Computational Linguistics. """
tag_set = set(['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS'])
"""
NN : 명사
NNP : 고유명사
JJ : 형용사
JJR, JJS : 형용사 비교급, 최상급
"""

iso_639_1 = {'en': 'english',
             'ar': 'arabic',
             'da': 'danish',
             'nl': 'dutch',
             'fi': 'finnish',
             'fr': 'french',
             'de': 'german',
             'hu': 'hungarian',
             'it': 'italian',
             'no': 'norwegian',
             'pt': 'portuguese',
             'ro': 'romanian',
             'ru': 'russian',
             'es': 'spanish',
             'sv': 'swedish'}

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def preprocessor(text):
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[\W]+', ' ', text.lower())
    return text


class NoneStemmer:
    """ Return work provided. Used if no stemmer in nltk for a language"""
    def stem(self, word):
        return word.lower()


class TopicRank:
    def __init__(self, text, lang='en'):
        # mapping of keyphrases to its positions in text
        self.phrases = defaultdict(list)
        self.unstem_map = {}
        # need to know the language for stemming
        # self.language = detect(text)
        self.language = lang
        logger.debug('Detected {} language'.format(self.language))
        # use stemmer if language is supported by nltk
        self.stemmer = SnowballStemmer(iso_639_1[self.language]) if self.language in iso_639_1 else NoneStemmer()
        logger.debug('Using {} stemmer'.format(self.stemmer.__class__))
        stop = stopwords.words(iso_639_1[self.language])
        self.text = []
        for word in sent_tokenize(text):
            # for word in word_tokenize(sent):
            self.text.append(word)

        self.topics = []

    def _extract_phrases(self):
        phrases = [[]]
        positions = []
        counter = 0
        for word, pos in nltk.pos_tag(self.text):
            if pos in tag_set:
                stemmed_word = self.stemmer.stem(word)
                if stemmed_word and len(stemmed_word) > 1:
                    phrases[-1].append(stemmed_word)
                    self.unstem_map[stemmed_word] = (counter, word)
                if len(phrases[-1]) == 1:
                    positions.append(counter)
            else:
                if phrases[-1]:
                    phrases.append([])
            counter += 1
        for n, phrase in enumerate(phrases):
            if phrase:
                self.phrases[' '.join(sorted(phrase))] = [i for i, j in enumerate(phrases) if j == phrase]
        logger.debug('Found {} keyphrases'.format(len(self.phrases)))

    def calc_distance(self, topic_a, topic_b):
        """
        Calculate distance between 2 topics
        :param topic_a: list if phrases in a topic A
        :param topic_b: list if phrases in a topic B
        :return: int
        """
        result = 0
        for phrase_a in topic_a:
            for phrase_b in topic_b:
                if phrase_a != phrase_b:
                    phrase_a_positions = self.phrases[phrase_a]
                    phrase_b_positions = self.phrases[phrase_b]
                    for a, b in product(phrase_a_positions, phrase_b_positions):
                        result += 1 / abs(a - b)
        return result

    def _identify_topics(self, strategy='average', max_d=0.75):
        """
        Group keyphrases to topics using Hierarchical Agglomerative Clustering (HAC) algorithm
        :param strategy: linkage strategy supported by scipy.cluster.hierarchy.linkage
        :param max_d: max distance for cluster identification using distance criterion in scipy.cluster.hierarchy.fcluster
        :return: None
        """
        # use term freq to convert phrases to vectors for clustering
        count = CountVectorizer()
        bag = count.fit_transform(list(self.phrases.keys()))

        # apply HAC
        Z = linkage(bag.toarray(), strategy)
        c, coph_dists = cophenet(Z, pdist(bag.toarray()))
        if c < 0.8:
            logger.warning("Cophenetic distances {} < 0.8".format(c))

        # identify clusters
        clusters = fcluster(Z, max_d, criterion='distance')
        cluster_data = defaultdict(list)
        for n, cluster in enumerate(clusters):
            inv = count.inverse_transform(bag.toarray()[n])
            cluster_data[cluster].append(' '.join(sorted([str(i) for i in count.inverse_transform(bag.toarray()[n])[0]])))
        logger.debug('Found {} keyphrase clusters (topics)'.format(len(cluster_data)))
        topic_clusters = [frozenset(i) for i in cluster_data.values()]
        # apply pagerank to find most prominent topics
        # Sergey Brin and Lawrence Page. 1998.
        # The Anatomy of a Large - Scale Hypertextual Web Search Engine.
        # Computer Networks and ISDN Systems 30(1): 107–117
        topic_graph = nx.Graph()
        topic_graph.add_weighted_edges_from(
            [(v, u, self.calc_distance(v, u)) for v in topic_clusters for u in topic_clusters if u != v])
        pr = nx.pagerank(topic_graph, weight='weight')

        # sort topic by rank
        self.topics = sorted([(b, list(a)) for a, b in pr.items()], reverse=True)

    def get_top_n(self, n=1, cluster_strategy='average', max_d=1.25, extract_strategy='first'):
        """
        Get topN topic based n ranks and select
        :param n: topN
        :param strategy: How to select keyphrase from topic:
                         -first - use the one which appears first
                         -center - use the center of the cluster WIP
                         -frequent - most frequent WIP
        :return: list of most ranked keyphrases
        """
        result = []
        self._extract_phrases()
        self._identify_topics(strategy=cluster_strategy, max_d=max_d)
        if extract_strategy != 'first':
            logger.warning("Using 'first' extract_strategy to extract keyphrases")
        for rank, topic in self.topics[:n]:
            if topic:
                first_kp = topic[0] #sorted(topic, key=lambda x: self.phrases[x][0])[0]
                unstem_kp_sort = sorted([self.unstem_map[i] for i in first_kp.split(' ')])
                unstem_kp = ' '.join([i[1] for i in unstem_kp_sort])
                result.append(unstem_kp)
        return result
