import csv
import sys
import string
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from .Dataset import Dataset
from scipy import stats


class DatasetEmoThreat (Dataset):
    """
    DatasetEmoThreat
    
    Multi-label emotion detection in the text has a lot of significance in both 
    research and industry for multiple applications of artificial intelligence. 
    Social media text can evoke multiple emotions in a small chunk of text, while 
    there is a possibility that text could be emotionless or neutral, making it a 
    challenging problem to tackle. 

    Urdu is spoken by more than 170 million people worldwide as a first and second 
    language, including India, Pakistan and Nepal. Needless to say that Urdu is 
    also widely used on social media using the right to left Nastalīq script. 
    Therefore, a multi-label emotion dataset for Urdu was long due and needed for 
    understanding public emotions, especially applicable in natural language 
    applications in disaster management, public policy, commerce, and public health. 

    We created a Nastalīq Urdu script dataset for multi-label emotion classification, 
    consisting of Twitter tweets using Ekman's six basic emotions and neutrality. 
    The task requires you to classify the tweet as one, or more of the six basic 
    emotions which is the best representation of the emotion of the person tweeting. 

    The task requires you to classify the tweet as one, or more of the six basic 
    emotions which is the best representation of the emotion of the person tweeting.

    - Anger: also includes annoyance and rage can be categorized as a response to 
      a deliberate attempt of anticipated danger, hurt or incitement.

    - Disgust: in the text is an inherent response of dis-likeness, loathing or 
      rejection to contagiousness.

    - Fear: also including anxiety, panic and horror is an emotion in a text 
      which can be seen triggered through a potential cumbersome situation or danger.

    - Sadness: also including pensiveness and grief is triggered through hardship, 
      anguish, feeling of loss, and helplessness.

    - Surprise: also including distraction and amazement is an emotion which is 
      prompted by an unexpected occurrence.

    - Happiness: also including contentment, pride, gratitude and joy is an emotion 
      which is seen as a response to well-being, a sense of achievement, satisfaction, and pleasure.

    - Neutral: is a tweet that does not evoke any emotion. 
    
    @link https://codalab.lisn.upsaclay.fr/competitions/5718
    
    @extends Dataset
    """

    def __init__ (self, dataset, options, corpus = '', task = '', refresh = False):
        """
        @inherit
        """
        Dataset.__init__ (self, dataset, options, corpus, task, refresh)
    
    def compile (self):

        # @var multiclass_labels List
        multiclass_labels = [
            'anger', 
            'disgust', 
            'fear', 
            'sadness', 
            'surprise', 
            'happiness', 
            'neutral'
        ]
        
        
        # Load dataframes
        dfs = []
        for index, dataframe in enumerate (['train', 'test']):
            
            # Open data
            df_split = pd.read_csv (self.get_working_dir ('dataset', dataframe + '.csv'))
            
            
            # Determine split
            df_split = df_split.assign (__split = 'test' if dataframe in 'test' else 'train')
            
            
            # Merge
            dfs.append (df_split)
        
        
        # Concat and assign
        df = pd.concat (dfs, ignore_index = True)


        # Rename columns
        df = df.rename (columns = {'Sentences': 'tweet'})
        
        
        # Individual traits
        for trait in multiclass_labels:
            df[trait] = df[trait].fillna (0.0)
            df[trait] = df[trait].astype ('bool')
            df.loc[df[trait] == True, trait] = trait
            df.loc[df[trait] == False, trait] = 'non-' + trait
        
        
        # Create the multi-label using the ";" separator
        df = df.assign (label = np.nan)
        df['label'] = df[multiclass_labels].agg ('; '.join, axis = 1)
        
        for trait in multiclass_labels:
            df['label'] = df['label'].str.replace ('non\-' + trait + ';?', '', regex = True)
        df['label'] = df['label'].str.replace ('; $', '', regex = True)
        df['label'] = df['label'].str.replace (';\s+', '; ', regex = True)
        df['label'] = df['label'].str.strip ()
        df['label'] = df['label'].str.strip (to_strip = ';')
        
        df = df.drop (multiclass_labels, axis = 1)
        
        
        # @var grouped_df Dataframe Randomized for getting the validation split
        grouped_df = df[df['__split'] == 'train'].sample (frac = 1)
        
        
        # @var grouped_labels Dataframe Randomized for getting the validation split
        grouped_labels = grouped_df.groupby ('label')
        
        
        # Next, we are going to perform a stratification split for generating the 
        # training and validation splits
        # ---------------------------------------------------------------------
        # @var task_items Sample validation test from training for task 2
        task_items = [np.split (g, [int(.2 * len (g))]) for i, g in grouped_labels]

        
        # @var training_indexes Sample validation test
        training_indexes = pd.concat ([t[0] for t in task_items])


        # Get the splits
        df.loc[training_indexes.index, '__split'] = 'val'
        df.loc[df['__split'] == 'test', '__split'] = 'test'
        
        
        # Rearrange labels
        df = df[['__split', 'label', 'tweet']]
        
        
        # Store this data on disk
        self.save_on_disk (df)
        
        
        # Return
        return df
        