# coding=utf-8

"""
整理原始数据
"""

from pandas import read_csv
from datetime import datetime


def parse(x):
    return datetime.strptime(x, "%Y %m %d %H")


dataset = read_csv("data/PRSA_data_2010.1.1-2014.12.31.csv", parse_dates=[['year', 'month', 'day', 'hour']],
                   index_col=0, date_parser=parse)

dataset.drop('No', axis=1, inplace=True)

dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'

dataset['pollution'].fillna(0, inplace=True)
dataset = dataset[24:]

print(dataset)
dataset.to_csv('data/pollution.csv')