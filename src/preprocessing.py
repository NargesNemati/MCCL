import pandas as pd 
from tqdm import tqdm

from collections import defaultdict

def convert_newid(origin_id:int, id_dict:dict, max_id:int):

    if origin_id in id_dict :
        new_id = id_dict.get(origin_id)
    else:
        id_dict[origin_id] = max_id
        new_id = max_id
        max_id += 1

    return new_id, id_dict, max_id


def reset_id(df, user_col='user_id', item_col='item_id', cols=['user_id', 'item_id', 'review', 'text_clean', 'ts', 'rating']):

    user_id_dict, item_id_dict, user_ids, item_ids, user_id_max, item_id_max  = {}, {}, [], [], 0, 0        
    
    for i in tqdm(range(len(df))):
        origin_user_id = df[user_col].iloc[i]
        origin_item_id = df[item_col].iloc[i]
        new_user_id, user_id_dict, user_id_max = convert_newid(origin_user_id, user_id_dict, user_id_max)
        new_item_id, item_id_dict, item_id_max = convert_newid(origin_item_id, item_id_dict, item_id_max)

        user_ids.append(new_user_id)
        item_ids.append(new_item_id)

    df[user_col] = user_ids
    df[item_col] = item_ids

    df = df[cols]
    return df
    


def get_cores(df):
    user_reviews = df.groupby('user_id').count()
    user_reviews = user_reviews.query('item_id >= 5')
    core_user = set(user_reviews.index)

    item_reviews = df.groupby('item_id').count()
    item_reviews = item_reviews.query('user_id >= 5')
    core_item = set(item_reviews.index)

    print(len(core_user), len(core_item))

    review_dic=defaultdict(list)
    for i, row in df.iterrows():
        user_id = row.user_id
        item_id = row.item_id
        rating = row.rating
        unix_time = row.unixReviewTime
        review = row.reviewText
        text_clean = row.text_clean
        if (user_id in core_user) and (item_id in core_item):
            review_dic['user_id'].append(user_id)
            review_dic['item_id'].append(item_id)
            review_dic['review'].append(review)
            review_dic['text_clean'].append(text_clean)
            review_dic['ts'].append(unix_time)
            review_dic['rating'].append(rating)

    core_df = pd.DataFrame(review_dic)
    return core_df


if __name__=='__main__':
    data_name='movie'
    
    df = pd.read_csv(f'data/{data_name}/{data_name}_clean.csv', index_col=0)
    df = df.dropna()
    df.drop_duplicates(keep='last', inplace=False)
    df = get_cores(df)
    df = reset_id(df)
    df.to_csv(f'{data_name}/{data_name}_fin.csv')