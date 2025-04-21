
import pandas as pd



if __name__=='__main__':
    data_name = 'movie'
    data_path=f'../data/{data_name}/{data_name}'
    train_df = pd.read_csv(f'{data_path}_train.csv')
    valid_df = pd.read_csv(f'{data_path}_valid.csv')
    test_df = pd.read_csv(f'{data_path}_test.csv')

    #accumulate
    valid_df = pd.concat([train_df, valid_df])
    all_df = pd.concat([valid_df, test_df])
    test_for_sh = pd.concat([valid_df, test_df])
    # test_for_sh = pd.read_csv(f'{data_path}.csv')
    shuffled_df = test_for_sh.sample(frac=1, random_state=42).reset_index(drop=True)
    test_len = int(len(shuffled_df) / 5)
    train_len = test_len * 3
    print(data_path)
    shuffled_df.to_csv(f'{data_path}.csv', index=False)


    # train_df = shuffled_df.head(train_len)
    # valid_df = shuffled_df.iloc[train_len+1:train_len+test_len]
    # test_df = shuffled_df.iloc[train_len+test_len+1:]
    # data_path=f'../data/{data_name}/{data_name}'

    # train_df.to_csv(f'{data_path}_train_new.csv', index=False)
    # valid_df.to_csv(f'{data_path}_valid_new.csv', index=False)
    # test_df.to_csv(f'{data_path}_test_new.csv', index=False)
