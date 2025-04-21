
from collections import defaultdict
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from topicrank import TopicRank
from tqdm import tqdm

class KeywordExtractor:
    def __init__(self, docs, n_gram_range=(1, 1)) -> None:
        self.n_gram_range = n_gram_range
        self.docs = docs
        # self.tf = self.get_tf()
        # self.word_count_df = pd.DataFrame({'count':(self.tf > 0).sum().sort_values()})

    def get_tf(self,):
        self.vect = CountVectorizer(ngram_range=self.n_gram_range,)
        document_term_matrix = self.vect.fit_transform(self.docs)
        tf = pd.DataFrame(document_term_matrix.toarray(), columns=self.vect.get_feature_names()) 
        return tf

    def _extract_doc_keywords(self,):
        raise NotImplementedError("extract_keywords 메소드를 구현하여야 합니다.")

    def extract_keywords(self, top_n=5) -> None:
        self.top_n = top_n
        doc_ids = []
        doc_kwds = []
        print(self.docs)

        for item in tqdm(self.docs.items()):            
            doc_id, doc = item
            doc_kwd = self._extract_doc_keywords(doc)
            doc_kwds.append(doc_kwd)
            doc_ids.append(doc_id)

        return pd.DataFrame({
            'doc_id' : doc_ids,
            'keywords' : doc_kwds
        })

    def get_keywords(self, duplicate_limit=0.1, num_keywords=512) -> set:
        # word_set = set(self.word_count_df.query('count>1').query(f'count<{duplicate_limit*len(self.docs)}').index.tolist())
        word_set = set()
        for kw, count in self.keywords.items():
            if count <= duplicate_limit*len(self.docs):
                word_set.add(kw)    
        filtered_kw = set()
        count = 0
        for w, s in sorted(self.keywords.items(), key=(lambda item: item[1]), reverse=True):
            if w in word_set:
                filtered_kw.add(w)
                count += 1
            if count == num_keywords:
                break
        return filtered_kw


class TFIDFExtractor(KeywordExtractor):
    def __init__(self, docs, n_gram_range=(1, 1)) -> None:
        super().__init__(docs, n_gram_range)
        self.stop_words = "english"
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words, smooth_idf=True, use_idf=True)
        self.vectorizer.fit_transform(self.docs)
        # self.feature_names = self.vectorizer.get_feature_names()
    
    def _sort_coo(self, coo_matrix):
        """Sort a dict with highest score"""
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def _extract_topn_from_vector(self, feature_names, sorted_items, topn=10):
        """get the feature names and tf-idf score of top n items"""
        sorted_items = sorted_items[:topn]
        score_vals = []
        feature_vals = []
        
        for idx, score in sorted_items:
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])

        results= {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]]=score_vals[idx]
        
        return results

    def _extract_doc_keywords(self, doc)->str:
        tf_idf_vector = self.vectorizer.transform([doc])
        sorted_items=self._sort_coo(tf_idf_vector.tocoo())
        keywords=self._extract_topn_from_vector(self.feature_names, sorted_items, self.top_n)
        doc_kwd = ''
        for kw, s in keywords.items():
            doc_kwd += kw + ' '
        return doc_kwd

        
class TopicRankExtractor(KeywordExtractor):
    def _extract_doc_keywords(self, doc)->str:
        tr = TopicRank(doc)
        try: keywords=tr.get_top_n(n=self.top_n)       
        except: keywords={}
        doc_kwd = ''
        for kw, s in keywords.items():
            doc_kwd += kw
        return doc_kwd

from summa import keywords

class TextRankExtractor(KeywordExtractor):

    def _extract_doc_keywords(self, doc)->str:
        keyword_list=keywords.keywords(doc).split('\n')[:self.top_n]       
        doc_kwd = ''
        for kw in keyword_list:
            doc_kwd += kw + ' '
        return doc_kwd


class KeyBertExtractor(KeywordExtractor):
    def __init__(self, docs, n_gram_range=(1, 1)) -> None:
        super().__init__(docs, n_gram_range)
        self.stop_words = "english"
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        # {word:total_freq}
        self.total_word_freq = defaultdict(int)

    def _extract_doc_keywords(self, doc)->str:
        try:
            count = CountVectorizer(ngram_range=self.n_gram_range, stop_words=self.stop_words).fit([doc])
        except:
            return ''
        candidates = count.get_feature_names_out()

        doc_embedding = self.model.encode([doc], show_progress_bar=False )
        candidate_embeddings = self.model.encode(candidates, show_progress_bar=False )

        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        keywords = [candidates[index] for index in distances.argsort()[0][-self.top_n:]]       
        doc_kwd = ''
        for kw in keywords:
            doc_kwd += kw + ' '
        return doc_kwd

    def get_tf_idkf(self,):
        docs_len = len(self.docs)
        kw_tfidkf_dict = dict()
        for kw, kw_count in self.keyword_count.items():
            tf = self.total_word_freq.get(kw, 0)
            kw_tfidkf_dict[kw] = tf * np.log(docs_len/kw_count)
        return kw_tfidkf_dict


class KeyBertEmbeddingExtractor(KeywordExtractor):
    def __init__(self, docs) -> None:
        super().__init__(docs)
        self.stop_words = "english"
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    
    def extract_embeddings(self,) -> None:
        embeddings=[]
        for doc in tqdm(self.docs):
            embedding = self.model.encode([doc])
            embeddings.append(embedding)
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings


# TODO
class LDAExtractor(KeywordExtractor):
    def extract_keywords(self, top_n=5):
        kws = defaultdict(int)
        for doc in tqdm(self.docs):
            keyword_list=keywords.keywords(doc).split('\n')[:top_n]
            for kw in keyword_list:
                kws[kw] += 1     
        self.keywords = kws 