
import pandas as pd
import numpy as np
import nltk
from collections import OrderedDict, defaultdict
import gensim
import gensim.corpora as corpora

from pprint import pprint

## train topic model with train docs 

## assign topic for all docs 

## build & save doc-topic vector (docs, topics)
#  
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

NUM_TOPICS = 128
th = 0.5


def get_topic_co_occurrence_matrix(node_topic_array):
    topic_co_occurrence_matrix = np.matmul(node_topic_array, node_topic_array.T)
    return  topic_co_occurrence_matrix

def get_topic_cosin_sim_matrix(node_topic_array):
    node_norm = node_topic_array.sum(axis=1)
    nrom_matrix = np.sqrt(np.outer(node_norm, node_norm))
    topic_co_occurrence_matrix = np.matmul(node_topic_array, node_topic_array.T)
    topic_cosin_sim_matrix = nrom_matrix*topic_co_occurrence_matrix
    return topic_cosin_sim_matrix

if __name__=='__main__':
    # load docs 
    for data_name in ['game', 'movie',
                      'music', 'book', 'office', 'sports', 'toy'
                      ]:
        train_df = pd.read_csv(f'data/{data_name}/{data_name}_train.csv')
        valid_df = pd.read_csv(f'data/{data_name}/{data_name}_valid.csv')
        test_df = pd.read_csv(f'data/{data_name}/{data_name}_test.csv')
        
        df = pd.concat([train_df, valid_df,])
        df = df.dropna()
        df['item_id'] += max(df.user_id)+1

        item_docs = df.groupby('item_id')['text_clean'].apply(lambda x: ' '.join(x))
        user_docs = df.groupby('user_id')['text_clean'].apply(lambda x: ' '.join(x))
        docs = item_docs.to_list() + user_docs.to_list()

        # data = df.text_clean.values.tolist()

        docs = [ str(d).split() for d in docs]
        id2word = corpora.Dictionary(docs)
        data_words = list(sent_to_words(docs))
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in data_words]

        # Build LDA model
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=NUM_TOPICS)


        df = pd.concat([train_df, valid_df, test_df])
        df = df.dropna()
        df['item_id'] += max(df.user_id)+1

        item_docs = df.groupby('item_id')['text_clean'].apply(lambda x: ' '.join(x))
        user_docs = df.groupby('user_id')['text_clean'].apply(lambda x: ' '.join(x))
        docs = item_docs.to_list() + user_docs.to_list()
        test_docs = [ str(d).split() for d in docs]
        
        # Print the Keyword in the topics
        pprint(lda_model.print_topics())
        test_doc_words = list(sent_to_words(test_docs))
        corpus = [id2word.doc2bow(text) for text in test_doc_words]

        n_nodes = len(test_docs)
        topic_vectors = np.zeros([n_nodes, NUM_TOPICS], dtype=np.int8)

        for i, topic_list in enumerate(lda_model[corpus]):
            if i<15:
                print(i,'번째 문서의 topic 비율은',topic_list)
            for t, prob in topic_list:
                if prob > th:
                    topic_vectors[i,t] = 1
        
        # topic_cosin_sim_matrix = get_topic_cosin_sim_matrix(topic_vectors)
        # with open(f'data/{data_name}/lda_topic{NUM_TOPICS}_cossim_matrix.npy', 'wb') as f:
        #     np.save(f, topic_cosin_sim_matrix)

        topic_cooc_matrix = get_topic_co_occurrence_matrix(topic_vectors)
        with open(f'data/{data_name}/lda_topic{NUM_TOPICS}_cooc_matrix.npy', 'wb') as f:
            np.save(f, topic_cooc_matrix)
