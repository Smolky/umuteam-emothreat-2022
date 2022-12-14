"""
    Merge a dataset based on user
    
    This process is necessary when dealing with author-profiling tasks
    It's purpose, is to merge all texts and features grouped by 
    user
    
    However, all features were collected individually using the 
    compile.py script
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import config
import bootstrap
import numpy as np
import pandas as pd

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.FeatureResolver import FeatureResolver
from features.TokenizerTransformer import TokenizerTransformer
from tqdm import tqdm

def main ():
    
    # var parser
    parser = DefaultParser (description = 'Merge dataset and features by author. Needed for profiling tasks')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)
    
    
    # @var df Dataframe
    df = dataset.get ()
    df = dataset.getDFFromTask (args.task, df)
    
    
    # Select those rows that are not null
    df = df[df['__split'].notna()]
    
    
    # @var columns_to_group List
    columns_to_group = dataset.get_columns_to_group_by_user ()
    
    
    # @var group DataframeGroup Grouping by user and split
    group = df.groupby (by = columns_to_group, dropna = False, observed = True, sort = False)

    
    # @var df_users_by_split Dataframe a Dataframe with the user and split
    df_users_by_split = group[columns_to_group].agg (func = ['count'], as_index = False, observed = True).index.to_frame (index = False)
    
    
    # @var fields_to_merge List
    fields_to_merge = ['tweet', 'tweet_clean', 'tweet_clean_lowercase', 'tagged_pos', 'tagged_ner']
    
    
    # @var merged_fields List
    merged_fields = []
    
    
    # @var pbar 
    pbar = tqdm (df_users_by_split.iterrows (), total = df_users_by_split.shape[0], desc = "merging users")
    
    
    # Iterare over rows in a fancy way
    for index, row in pbar:
        
        # @var df_user DataFrame Select by user and split
        df_user = df[(df['user'] == row['user']) & (df['__split'] == row['__split'])]
        

        # Attach data
        merged_fields.append ({**row, **{field: ' [SEP] '.join (df_user[field].fillna ('')) for field in fields_to_merge}})
    
    
    
    # @var df_merged DataFrame
    df_merged = pd.DataFrame (merged_fields)


    
    # Store the merged dataset
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    dataset.save_on_disk (df_merged)

    
    # @var available_features List
    available_features = ['lf', 'be', 'se', 'bf', 'ng', 'rf']
    
    
    # @var pbar
    pbar = tqdm (available_features)
    
    
    # @var user_per_split Dict
    user_per_split = df_merged[['user', '__split']].to_dict (orient = 'records')
    
    
    
    # Get every feature set
    for feature_set in pbar:
    
        # @var prefixed List We get all the selected features subsets
        prefixed = [filename for filename in os.listdir (dataset.get_working_dir (args.task)) if filename.startswith (feature_set) and filename.endswith ('.csv')]
        
        
        # Iterate over each subset
        for feature_file in prefixed:
    
            # update pbar
            pbar.set_description (feature_file + " (...reading)")


            # @var transformer Transformer 
            transformer = feature_resolver.get (feature_set, dataset.get_working_dir (args.task, feature_file))
        

            # @var features_df DataFrame Get all the unmerged features 
            features_df = transformer.transform ([])
            
            
            print (features_df)
            
            
            # Skip. Features already merged
            if len (features_df) == len (df_merged):
                continue;
            
            
            # update pbar
            pbar.set_description (feature_file + " (...transforming)")
            
            
            # @var features_df_rows List Here we select subsets of the features grouped by the user and the split
            # and we calculate the mean of each column
            features_df_merged = pd \
                .concat ([features_df.loc[df.loc[(df['user'] == dict['user']) & (df['__split'] == dict['__split'])].index].mean ().to_frame ().T for dict in user_per_split]) \
                .reset_index (drop = True)

        
        
            # @var merged_features_path String
            merged_features_path = dataset.get_working_dir (args.task, feature_file)
            
            
            # update pbar
            pbar.set_description (feature_file + " (...saving)")

            
            # Store the features on disk
            features_df_merged.to_csv (merged_features_path, index = False, float_format = '%.10f')


    # @var df_merged_train Dataframe
    df_merged_train = dataset.get_split (df_merged, 'train')


    # @var we_transformers WE
    """
    we_transformers = TokenizerTransformer (cache_file = dataset.get_working_dir (args.task, 'we.csv'), field = 'tweet_clean_lowercase')
    
    
    # Fit over merged dataframe
    we_transformers.fit (df_merged_train)
    print (we_transformers.transform (df_merged))
    we_transformers.save_tokernizer_on_disk (dataset.get_working_dir (args.task, 'we_tokenizer.pickle'))
    """


if __name__ == "__main__":
    main ()