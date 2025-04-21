from collections import defaultdict
import pandas as pd
import pickle as pkl
from itertools import combinations, product
import gc
import time
from random import sample

EDGE_LIMIT = 1024

def get_id_keyword_dict(df)->defaultdict:
    keyword_id_dict = defaultdict(set)
    for _, row in df.iterrows():
        id = row.doc_id
        try: kws = row.keywords.split()
        except: kws = []
        for kw in kws:
            keyword_id_dict[kw].add(id)
    print('keyword id dict')
    return keyword_id_dict
    
def get_cooc_pair_df(keyword_id_dict)->pd.DataFrame:
    """
    return cooc edge as df format
    uid, vid, cooc_count
    """
    pair_set = set()
    us, vs = [], []
    for kw, ids in keyword_id_dict.items():
        # if len(ids) > 1024:
        #     ids = sample(ids, 1024)

        pairs = list(combinations(ids, 2))
        if len(pairs) > EDGE_LIMIT:
            pairs = sample(pairs, EDGE_LIMIT)
        for u, v in pairs:
            if u > v:
                u,v=v,u
            if (u,v) not in pair_set:
                pair_set.add((u,v))        
                us.append(u)
                vs.append(v)

    print('homo pair')
    return pd.DataFrame({
        'u':us,
        'v':vs,
        # 'c':cs,
    })


def get_uv_cooc_pair_df(u_dict, v_dict)->pd.DataFrame:
    """
    return cooc edge between `user-item` as df format
    uid, vid, cooc_count
    """
    pair_set = set()
    for kw, u_ids in u_dict.items():
        if kw not in v_dict:
            continue
        v_ids = v_dict.get(kw,set())
        pairs = list(product(u_ids, v_ids))
        if len(pairs) > EDGE_LIMIT:
            pairs = sample(pairs, EDGE_LIMIT)
        for u, v in pairs:
            if (u,v) not in pair_set:
                pair_set.add((u,v))

    us, vs, cs = [], [], []
    for u, v in pair_set:
        us.append(u)
        vs.append(v)
        # cs.append(c)

    print('user-item pair')
    return pd.DataFrame({
        'u':us,
        'i':vs,
        # 'c':cs,
    })


if __name__=='__main__':
    for keyword_extraction_method in [
                # 'text_rank',
                'keybert', 
                # 'tfidf',        
                # 'topic_rank',
        ]:
        for data_name in [
                        'tools',
                        # 'grocery',

                        # 'movie', 
                        # 'yelp',
                        # 'games', 
                        # 'music', 
                        # 'beauty',
                        # 'office', 
                        # 'sports',
                        ]:
            
            item_df = pd.read_csv(f'../../data/{data_name}/item_{keyword_extraction_method}_keywords.csv')
            user_df = pd.read_csv(f'../../data/{data_name}/user_{keyword_extraction_method}_keywords.csv')

            item_dict = get_id_keyword_dict(item_df)
            cooc_df = get_cooc_pair_df(item_dict)
            cooc_df.to_csv(f'../../data/{data_name}/item_{keyword_extraction_method}_cooc.csv')
            print('item : ', data_name, keyword_extraction_method,len(cooc_df))
            del cooc_df
            gc.collect()
            print('item', time.time())

            user_dict = get_id_keyword_dict(user_df)
            cooc_df = get_cooc_pair_df(user_dict)
            print('user : ', data_name, keyword_extraction_method,len(cooc_df))
            cooc_df.to_csv(f'../../data/{data_name}/user_{keyword_extraction_method}_cooc.csv')
            del cooc_df
            gc.collect()
            print('user', time.time())

            cooc_df = get_uv_cooc_pair_df(user_dict, item_dict)
            print('user-item : ', data_name, keyword_extraction_method,len(cooc_df))
            print('user-item', time.time())
            cooc_df.to_csv(f'../../data/{data_name}/user_item_{keyword_extraction_method}_cooc.csv')
