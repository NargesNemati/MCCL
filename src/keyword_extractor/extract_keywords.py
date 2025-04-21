
import pandas as pd
import numpy as np
import nltk
import pickle as pkl
from collections import OrderedDict, defaultdict

from keyword_extractor import KeyBertExtractor, TFIDFExtractor, TopicRankExtractor, \
                              TextRankExtractor, KeyBertEmbeddingExtractor

# def get_keyword_co_occurrence_matrix(nid_arr_dict):
#     nid_arr_dict = OrderedDict(sorted(nid_arr_dict.items()))
#     keyword_matrix = np.array([ item for k, item in nid_arr_dict.items() ])
#     keyword_co_occurrence_matrix = np.matmul(keyword_matrix, keyword_matrix.T)
    
#     return  keyword_co_occurrence_matrix

# def get_keyword_co_occurrence_dict(nid_arr_dict, user_item_pairs):
#     nid_arr_dict = OrderedDict(sorted(nid_arr_dict.items()))
#     keyword_matrix = np.array([ item for k, item in nid_arr_dict.items() ])
#     keyword_co_occurrence_dict = {}
#     for uid, iid in user_item_pairs:
#         user_matrix = keyword_matrix[uid]
#         item_matrix = keyword_matrix[iid]
#         key = str(uid)+'_'+str(iid)
#         keyword_co_occurrence_dict[key] = np.sum(user_matrix*item_matrix)
    
#     return  keyword_co_occurrence_dict

# def get_keyword_cosin_sim_matrix(nid_arr_dict):
#     nid_arr_dict = OrderedDict(sorted(nid_arr_dict.items()))
#     keyword_matrix = np.array([ item for k, item in nid_arr_dict.items() ])
#     node_norm = keyword_matrix.sum(axis=1)
#     nrom_matrix = np.sqrt(np.outer(node_norm, node_norm))
#     topic_co_occurrence_matrix = np.matmul(keyword_matrix, keyword_matrix.T)
#     topic_cosin_sim_matrix = nrom_matrix*topic_co_occurrence_matrix
#     return topic_cosin_sim_matrix

# def convert_doc_array(item_docs, user_docs, kw_df, large=False):
    
#     def get_keyword_vector(text:str, keywords_index_dict:dict):
#         n = len(keywords_index_dict)
#         one_hot_vector = [0]*n
#         words = text.split()
#         for w in words:
#             i = keywords_index_dict.get(w)
#             if i != None:
#                 one_hot_vector[i] = 1

#         return one_hot_vector
    

#     keywords_index_dict = {}
#     kw_id = 0 
#     kw_set = set(kw_df['keyword'])
#     for kw in kw_set:
#         if kw not in keywords_index_dict:
#             keywords_index_dict[kw] = kw_id
#             kw_id += 1

#     nid_arr_dict = {}
#     print(len(item_docs), len(user_docs))
#     for i, d in zip(item_docs.index, item_docs.values):
#         nid_arr_dict[i] = get_keyword_vector(d, keywords_index_dict)

#     for i, d in zip(user_docs.index, user_docs.values):
#         nid_arr_dict[i] = get_keyword_vector(d, keywords_index_dict)
    
#     return nid_arr_dict




if __name__=='__main__':
    TOP_K = 5

    # load docs 
    keyword_extractors = {
        'text_rank' : TextRankExtractor,
        'tfidf' : TFIDFExtractor,
        'topic_rank' : TopicRankExtractor,
        'keybert' : KeyBertExtractor,
    }

    for keyword_extraction_method in [
        # 'tfidf',
        'keybert',
        # 'text_rank',
        # 'topic_rank'
    ]:
        for data_name in [
                        # 'movie',
                        # 'yelp',
                        # 'grocery',
                        'tools',
                        # 'games', 
                        # 'beauty',
                        # 'music', 
                        # 'office', 
                        # 'sports',
                        ]:
            train_df = pd.read_csv(f'../../data/{data_name}/{data_name}_train_new.csv')
            # valid_df = pd.read_csv(f'../../data/{data_name}/{data_name}_valid.csv')
            # test_df = pd.read_csv(f'data/{data_name}/{data_name}_test.csv')

            # df = pd.concat([train_df, valid_df])
            df = train_df
            # df = df.dropna()
            
            review_col = 'reviewText'
            for r in df[review_col]:
                if type(r) != str:
                    print(r)

            item_docs = df.groupby('item_id')[review_col].apply(lambda x: ' '.join(str(i) for i in x if pd.notna(i)))
            user_docs = df.groupby('user_id')[review_col].apply(lambda x: ' '.join(str(i) for i in x if pd.notna(i)))

            # user_doc
            keyword_extractor_class = keyword_extractors.get(keyword_extraction_method)
            keyword_extractor = keyword_extractor_class(user_docs, n_gram_range=(1,1))
            user_df = keyword_extractor.extract_keywords(top_n=TOP_K)
            print(user_df[:10])
            user_df.to_csv(f'../../data/{data_name}/user_{keyword_extraction_method}_keywords.csv')

            # item_doc
            keyword_extractor = keyword_extractor_class(item_docs, n_gram_range=(1,1))
            item_df = keyword_extractor.extract_keywords(top_n=TOP_K)
            print(item_df[:10])
            item_df.to_csv(f'../../data/{data_name}/item_{keyword_extraction_method}_keywords.csv')
