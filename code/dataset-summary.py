"""
    Show dataset statistics for label and split distribution
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import config
import bootstrap
import os.path
import pandas as pd

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from utils.LabelsDistribution import LabelsDistribution
from utils.WordsCloud import WordsCloud
from utils.CorpusStatistics import CorpusStatistics


def main ():
    
    # var parser
    parser = DefaultParser (description = 'Show statistics from the datasets')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, args.force)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # @var df Dataframe
    df = dataset.get ()
    
    
    # @var split_field String
    split_field = '__split'
    
    
    # @var label_field String
    label_field = 'label'
    
    
    # @var grouping_per_user Boolean Determines if the statistics of the dataset 
    #                       should be grouped by tweet (default) or by 
    #                       group
    grouping_per_user = dataset.group_results_by_user ()
    
    
    # Keep train, validation, and test
    df = df.loc[~df[split_field].isna ()]
    df = df.loc[df[split_field].isin (['train', 'val', 'test'])]
    
    
    # @var corpus_statistics CorpusStatistics
    corpus_statistics = CorpusStatistics (dataset)
    
    
    # To avoid information
    with pd.option_context ('display.max_rows', 1000, 'display.max_columns', 10):
        
        # Information concerning the dates
        if 'twitter_created_at' in df.columns:

            # @var df_dates DataFrame Force date field is a date
            df_dates = df.copy ()
            df_dates['twitter_created_at'] = pd.to_datetime (df_dates['twitter_created_at'], errors = 'coerce')


            # @var df_dates_by_month DataFrame 
            df_dates_by_month = df_dates['twitter_created_at'].groupby (df_dates['twitter_created_at'].dt.to_period ('M')).agg ('count')
            
            
            # @var min_date String
            min_date = df_dates.set_index ('twitter_created_at').index.min ()
            
            
            # @var max_date String
            max_date = df_dates.set_index ('twitter_created_at').index.max ()
            

            print ()
            print ('dataset dates')
            print ('-------------')
            
            print ()
            print ('min and max dates')
            print (min_date)
            print (max_date)
            
            print ()
            print ('distribution')
            print (df_dates_by_month)


            # @var dates_by_month_path String
            dates_by_month_path = dataset.get_working_dir (dataset.task, 'statistics', 'date-distribution.csv')

            
            # Save in disk
            df_dates_by_month.to_csv (dates_by_month_path, index = False)


        
        if 'user' in df.columns:
            print ()
            print ('user distribution')
            print ('-----------------')
            
            # @var user_counts
            user_counts = df['user'].value_counts (sort = False)
            
            
            # @var how_many_different_users int
            how_many_different_users = len (user_counts)
            
            
            # If there are few users, we get some useful statistics
            if how_many_different_users >= 10:
                print ('The mean is {}'.format (user_counts.mean ()))
                print ('The median is {}'.format (user_counts.median ()))
                print ('The mode is {}'.format (user_counts.mode ()))
                print ('The standard deviation is {}'.format (user_counts.std ()))
            
            else:
                print (user_counts.sort_index ())
                print (df['user'].value_counts (normalize = True, sort = False).sort_index ())


        # @var split_df DataFrame
        split_df = pd.DataFrame ()
        split_df = split_df.assign (train = '')
        split_df = split_df.assign (val = '')
        split_df = split_df.assign (test = '')
        
        
        # @var splits List
        splits = ['train', 'val', 'test']
        

        # Get the stastics per split
        # Multiclassification
        if dataset.get_task_type () == 'classification' and not grouping_per_user:
            for split in splits:
                split_df[split] = dataset.get_split (df, split, split_field = split_field)[label_field].value_counts (sort = False).sort_index ()
        
        # Multilabel
        elif dataset.get_task_type () == 'multi_label' and not grouping_per_user:
            for split in splits:
                
                # @var df_distribution DataFrame
                df_distribution = dataset.get_split (df, split, split_field = split_field)
                
                
                # @var df_labels DataFrame
                df_labels = df_distribution[label_field].str.split ('; ', expand=True)
                
                
                # @var columns List
                columns = [df_labels[column] for column in df_labels]
                
                
                # Update labels
                df_labels = pd.concat (columns, axis = 0, ignore_index = True).dropna (axis = 'rows')
                df_distribution = df_labels.value_counts (sort = False).sort_index ()
                split_df[split] = df_distribution

        # Regression
        elif dataset.get_task_type () == 'regression':
            print (df.loc[df[label_field].notnull()][label_field].describe ())
            print (df.loc[df[label_field].notnull()][label_field].mode ())
            
            sys.exit ()
    
    
    # @var label_distribution_path String
    label_distribution_path = dataset.get_working_dir (dataset.task, 'statistics', 'label_distribution_per_split.latex')
    
    
    # Add totals in rows and columns
    split_df['total'] = split_df.sum (axis = 1)
    split_df.loc['total'] = split_df.sum (numeric_only = True, axis = 0)
    

    print (split_df)
    split_df.to_latex (label_distribution_path)

if __name__ == "__main__":
    main ()