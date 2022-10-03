import csv
import sys
import string
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from .Dataset import Dataset
from scipy import stats


class DatasetEmoThreatTask2 (Dataset):
    """
    DatasetEmoThreatTask2

    @link https://codalab.lisn.upsaclay.fr/competitions/5718
    
    @extends Dataset
    """

    def __init__ (self, dataset, options, corpus = '', task = '', refresh = False):
        """
        @inherit
        """
        Dataset.__init__ (self, dataset, options, corpus, task, refresh)
    
    def compile (self):
        
        # Load dataframes
        dfs = []
        for index, dataframe in enumerate (['train', 'test']):
            
            # Open data
            df_split = pd.read_excel (self.get_working_dir ('dataset', dataframe + '.xlsx'))
            
            
            # Determine split
            df_split = df_split.assign (__split = 'test' if dataframe in 'test' else 'train')
            
            
            if dataframe == 'test':
                df_split = df_split.rename (columns = {'Tweet': 'Tweets'})
            

            # Merge
            dfs.append (df_split)
        
        
        # Concat and assign
        df = pd.concat (dfs, ignore_index = True)


        # Rename columns
        df = df.rename (columns = {'Tweets': 'tweet', 'S/G': 'target'})
        

        # Reassign labels
        df.loc[df['label'] == True, 'label'] = 'threatening'
        df.loc[df['label'] == False, 'label'] = 'non-' + 'threatening'

        df.loc[df['target'] == 0, 'target'] = 'individual'
        df.loc[df['target'] == 1, 'target'] = 'group'
        df.loc[df['target'] == 2, 'target'] ='non-' + 'threatening'
        
        
        # Rearrange labels
        df = df[['__split', 'label', 'target', 'tweet']]


        # @var training_indexes Sample validation test
        training_indexes = df[df['__split'] == 'train'].sample (frac = 0.2)
        df.loc[training_indexes.index, '__split'] = 'val'
        
        
        # Store this data on disk
        self.save_on_disk (df)
        
        
        # Return
        return df
        