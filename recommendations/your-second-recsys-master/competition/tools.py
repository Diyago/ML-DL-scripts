import random
import datetime
import numpy as np

import scipy.sparse as sp
import pandas as pd

from itertools import islice, cycle
from more_itertools import pairwise
from implicit.nearest_neighbours import TFIDFRecommender, BM25Recommender, CosineRecommender


class TimeRangeSplit():
    """
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html
    """
    def __init__(self, 
                 start_date, 
                 end_date=None, 
                 freq='D', 
                 periods=None, 
                 tz=None, 
                 normalize=False, 
                 closed=None, 
                 train_min_date=None,
                 filter_cold_users=True, 
                 filter_cold_items=True, 
                 filter_already_seen=True):
        
        self.start_date = start_date
        if end_date is None and periods is None:
            raise ValueError('Either "end_date" or "periods" must be non-zero, not both at the same time.')

        self.end_date = end_date
        self.freq = freq
        self.periods = periods
        self.tz = tz
        self.normalize = normalize
        self.closed = closed
        self.train_min_date = pd.to_datetime(train_min_date, errors='raise')
        self.filter_cold_users = filter_cold_users
        self.filter_cold_items = filter_cold_items
        self.filter_already_seen = filter_already_seen

        self.date_range = pd.date_range(
            start=start_date, 
            end=end_date, 
            freq=freq, 
            periods=periods, 
            tz=tz, 
            normalize=normalize, 
            closed=closed)

        self.max_n_splits = max(0, len(self.date_range) - 1)
        if self.max_n_splits == 0:
            raise ValueError('Provided parametrs set an empty date range.') 

    def split(self, 
              df, 
              user_column='user_id',
              item_column='item_id',
              datetime_column='date',
              fold_stats=False):
        df_datetime = df[datetime_column]
        if self.train_min_date is not None:
            train_min_mask = df_datetime >= self.train_min_date
        else:
            train_min_mask = df_datetime.notnull()

        date_range = self.date_range[(self.date_range >= df_datetime.min()) & 
                                     (self.date_range <= df_datetime.max())]

        for start, end in pairwise(date_range):
            fold_info = {
                'Start date': start,
                'End date': end
            }
            train_mask = train_min_mask & (df_datetime < start)
            train_idx = df.index[train_mask]
            if fold_stats:
                fold_info['Train'] = len(train_idx)

            test_mask = (df_datetime >= start) & (df_datetime < end)
            test_idx = df.index[test_mask]
            
            if self.filter_cold_users:
                new = np.setdiff1d(
                    df.loc[test_idx, user_column].unique(), 
                    df.loc[train_idx, user_column].unique())
                new_idx = df.index[test_mask & df[user_column].isin(new)]
                test_idx = np.setdiff1d(test_idx, new_idx)
                test_mask = df.index.isin(test_idx)
                if fold_stats:
                    fold_info['New users'] = len(new)
                    fold_info['New users interactions'] = len(new_idx)

            if self.filter_cold_items:
                new = np.setdiff1d(
                    df.loc[test_idx, item_column].unique(), 
                    df.loc[train_idx, item_column].unique())
                new_idx = df.index[test_mask & df[item_column].isin(new)]
                test_idx = np.setdiff1d(test_idx, new_idx)
                test_mask = df.index.isin(test_idx)
                if fold_stats:
                    fold_info['New items'] = len(new)
                    fold_info['New items interactions'] = len(new_idx)

            if self.filter_already_seen:
                user_item = [user_column, item_column]
                train_pairs = df.loc[train_idx, user_item].set_index(user_item).index
                test_pairs = df.loc[test_idx, user_item].set_index(user_item).index
                intersection = train_pairs.intersection(test_pairs)
                print(f'Already seen number: {len(intersection)}')
                test_idx = test_idx[~test_pairs.isin(intersection)]
                # test_mask = rd.df.index.isin(test_idx)
                if fold_stats:
                    fold_info['Known interactions'] = len(intersection)

            if fold_stats:
                fold_info['Test'] = len(test_idx)

            yield (train_idx, test_idx, fold_info)

    def get_n_splits(self, df, datetime_column='date'):
        df_datetime = df[datetime_column]
        if self.train_min_date is not None:
            df_datetime = df_datetime[df_datetime >= self.train_min_date]

        date_range = self.date_range[(self.date_range >= df_datetime.min()) & 
                                     (self.date_range <= df_datetime.max())]

        return max(0, len(date_range) - 1)
    
    
    
def get_coo_matrix(df, 
                   user_col='user_id', 
                   item_col='item_id', 
                   weight_col=None, 
                   users_mapping={}, 
                   items_mapping={}):
    
    if weight_col is None:
        weights = np.ones(len(df), dtype=np.float32)
    else:
        weights = df[weight_col].astype(np.float32)

    interaction_matrix = sp.coo_matrix((
        weights, 
        (
            df[user_col].map(users_mapping.get), 
            df[item_col].map(items_mapping.get)
        )
    ))
    return interaction_matrix


def generate_implicit_recs_mapper(
    model,
    train_matrix,
    top_N,
    user_mapping,
    item_inv_mapping,
    filter_already_liked_items
):
    def _recs_mapper(user):
        user_id = user_mapping[user]
        recs = model.recommend(user_id, 
                               train_matrix, 
                               N=top_N, 
                               filter_already_liked_items=filter_already_liked_items)
        return [item_inv_mapping[item] for item, _ in recs]
    return _recs_mapper


def generate_recs(model, mapper, list_user_id):
    recs = pd.DataFrame({'user_id': list_user_id})
    recs['item_id'] = recs['user_id'].map(mapper)
    recs = recs.explode('item_id')
    recs['rank'] = recs.groupby('user_id').cumcount() + 1
    return recs

def compute_metrics(train, test, recs, top_N):
    result = {}
    test_recs = test.set_index(['user_id', 'item_id']).join(recs.set_index(['user_id', 'item_id']))
    test_recs = test_recs.sort_values(by=['user_id', 'rank'])

    test_recs['users_item_count'] = test_recs.groupby(level='user_id')['rank'].transform(np.size)
    test_recs['reciprocal_rank'] = (1 / test_recs['rank']).fillna(0)
    test_recs['cumulative_rank'] = test_recs.groupby(level='user_id').cumcount() + 1
    test_recs['cumulative_rank'] = test_recs['cumulative_rank'] / test_recs['rank']
    
    users_count = test_recs.index.get_level_values('user_id').nunique()
    
    result[f'MAP@{top_N}'] = (test_recs['cumulative_rank'] / test_recs['users_item_count']).sum() / users_count
    
    return pd.Series(result)

def get_ranks_sum(recs_with_ranks, intersection):
    cumulative_ranks_total = 0
    
    for idx, item_id in enumerate(intersection):
        if recs_with_ranks.get(item_id, None):
            cumulative_ranks_total += (idx+1)/recs_with_ranks.get(item_id)
    return cumulative_ranks_total

def fillna_cols(df):
    return df.fillna({'age':'age_unknown','sex':'sex_unknown', 'income': 'income_unknown','kids_flg': False}).copy()

def compute_metric_leaderboard(test, recs):
    recs_g = recs.reset_index()
    test_g = test.reset_index()
    recs_g.item_id = recs_g.item_id.apply(lambda x: x.strip("[]").split(', '))
    test_g.item_id = test_g.item_id.apply(lambda x: x.strip("[]").split(', '))
    
    test_recs_g = pd.merge(test_g, recs_g, on='user_id', how='left')
    test_recs_g['item_id_y'] = test_recs_g['item_id_y'].fillna('').apply(list)
    test_recs_g['users_item_count'] = test_recs_g['item_id_x'].apply(len)
    test_recs_g['recs_with_ranks'] = test_recs_g['item_id_y'].apply(lambda x: {vl: idx+1 for idx, vl in enumerate(x)})
    test_recs_g['intersection'] = [[i for i in a if str(i) in str(b)] for a, b in zip(test_recs_g.item_id_y, test_recs_g.item_id_x)]
    test_recs_g['cumulative_ranks_total'] = test_recs_g.apply(
        lambda x: get_ranks_sum(x['recs_with_ranks'], x['intersection']), axis=1)
    test_recs_g['cumulative_ranks_total'] = test_recs_g['cumulative_ranks_total'] / test_recs_g['users_item_count']
    users_count = test_recs_g.user_id.nunique()
    return test_recs_g['cumulative_ranks_total'].sum()/users_count