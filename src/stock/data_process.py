# coding=utf-8

import tushare as ts
import pandas as pd
from zibiao.zb import ZB


def _fix_p_change(st_df):
    st_df['p_change'] = 0
    st_df['p_change'] = ((st_df['close'] - st_df['close'].shift(1)) / st_df['close'].shift(1) * 100).round(2)


def get_k_data():
    st_df = ts.get_k_data("000001", start="2000-01-01")

    st_df.set_index("date", inplace=True)
    st_df.sort_index(ascending=True, inplace=True)
    _fix_p_change(st_df)
    return st_df


def process_data(st_df):
    vol_df = ZB.vol(st_df)

    st_df['p_change2'] = st_df['p_change'].rolling(center=False, min_periods=1, window=2).sum()
    st_df['p_change5'] = st_df['p_change'].rolling(center=False, min_periods=1, window=5).sum()
    st_df['p_change10'] = st_df['p_change'].rolling(center=False, min_periods=1, window=10).sum()

    st_df['next_p_change2'] = st_df['p_change2'].shift(-2)

    st_df['vma3_10'] = vol_df['VMA3'] / vol_df['VMA10']
    st_df['vma5_10'] = vol_df['VMA5'] / vol_df['VMA10']

    st_df.dropna(inplace=True)

    st_df['target'] = 'D'

    st_df.ix[(st_df['next_p_change2']< -10), 'target'] = 'A'
    st_df.ix[(st_df['next_p_change2'] >= -10) & (st_df['next_p_change2'] < -5), 'target'] = 'B'
    st_df.ix[(st_df['next_p_change2'] >= -5) & (st_df['next_p_change2'] < -2), 'target'] = 'C'
    st_df.ix[(st_df['next_p_change2'] >= -2) & (st_df['next_p_change2'] <= 0), 'target'] = 'D'

    st_df.ix[(st_df['next_p_change2'] > 0) & (st_df['next_p_change2'] <= 2), 'target'] = 'E'
    st_df.ix[(st_df['next_p_change2'] > 2) & (st_df['next_p_change2'] <= 5), 'target'] = 'F'
    st_df.ix[(st_df['next_p_change2'] > 5) & (st_df['next_p_change2'] <= 10), 'target'] = 'G'
    st_df.ix[(st_df['next_p_change2'] > 10), 'target'] = 'H'

    names = ['p_change', 'p_change2', 'p_change5', 'p_change10', 'vma3_10', 'vma5_10', 'target']
    return st_df.ix[:, names]


if __name__ == '__main__':

    st_df = get_k_data()
    target_df = process_data(st_df)
    print(target_df)
