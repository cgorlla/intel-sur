'''
etl.py contains functions used to load in data collected from Input Libraries.
'''

import os
import pandas as pd
import numpy as np
import sqlite3
from collections import defaultdict
from sklearn.model_selection import train_test_split


def join_tables(tables):
    '''
    Helper function to join tables across different .db files
    '''
    
    df = tables[0]
    for table in tables[1:]:
        df = pd.concat([df, table])
        
    return df.reset_index(drop=True)


def get_all_databases(folder):
    '''
    Joins all files in a data folder and returns relevant dataframes
    '''
    string_tables = []
    ull_tables = []
    files = os.listdir(folder)
    
    # iterating through all files of folder to append tables
    for file in files:
        path = os.path.join(folder, file)
        con = sqlite3.connect(path)
        string_tables.append(pd.read_sql_query('SELECT * FROM COUNTERS_STRING_TIME_DATA', con))
        ull_tables.append(pd.read_sql_query('SELECT * FROM COUNTERS_ULL_TIME_DATA', con))
        
    # concatenating tables into single dataframe
    string_df = join_tables(string_tables)
    string_df.loc[:, 'VALUE'] = string_df.loc[:, 'VALUE'] 
    string_df.loc[:, 'VALUE'] = string_df.loc[:, 'VALUE'].str.lower()
    ull_df = join_tables(ull_tables)
    
    return (string_df, ull_df)
