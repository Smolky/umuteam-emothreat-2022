"""
    LF Histogram
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import re
import argparse
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

from tqdm import tqdm 
from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.FeatureResolver import FeatureResolver


def main ():

    # Set theme
    sn.set_theme ()
    sn.set_palette ('pastel')


    # @var color_palette List
    color_palette = sn.color_palette ('pastel').as_hex ()


    # @var parser
    parser = DefaultParser (description = 'Generate an histogram of the linguistic features')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, args.force)
    
    
    # @var df Ensure if we already had the data processed
    df = dataset.get ()
    
    
    # @var train_df DataFrame Get training split
    train_df = dataset.get_split (df, 'train')
    
    
    # @var indexes Dict the the indexes for each split
    indexes = {split: subset.index for split, subset in {'train': train_df}.items ()}
    
    
    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)
    
    
    # @var features_cache String Retrieve 
    features_cache = dataset.get_working_dir ('lf.csv')
    
        
    # @var transformer
    transformer = feature_resolver.get ('lf', features_cache)
    
    
    # @var features_df DataFrame
    features_df = transformer.transform ([]);
    
    
    # Get the features from the train split
    features_df = features_df[features_df.index.isin (indexes['train'])].reindex (indexes['train'])


    # Remove columns that are all zeros
    features_df.loc[:, (features_df != 0).any (axis = 0)]


    # @var column_index int
    column_index = 0
    
    
    # @var n_columns int
    n_columns = 20
    
    
    # @var n_rows int
    n_rows = len (features_df.columns) // n_columns
    
    
    # @var width_in_inches float
    width_in_inches = 5
    
    
    # @var height_in_inches float
    height_in_inches = 5
    
    
    # @var bins int
    bins = 10
    
    
    # @var categories List
    categories = list (set ([column.split ('-')[0] for column in features_df.columns if '-' in column])) 
    
    
    # Rename column names
    column_names = list (features_df.columns)
    column_names = [column.replace ('-', '\n', 1) for column in column_names]

    
    # @var fig, ax
    fig, ax = plt.subplots (n_rows, n_columns, 
        sharex = False, 
        sharey = False,
        figsize = (width_in_inches * n_columns, height_in_inches * n_rows)
    )
    
    
    # @var pbar
    pbar = tqdm (range (n_rows))
    
    
    # Update names
    features_df.columns = column_names
    
    
    # Generate the subplots
    for i in pbar:
        for j in range (n_columns):
            
            # Update progress bar
            pbar.set_description ("Processing %s (%s, %s)" % (column_index, i, j))
            
            
            # @var column Series 
            column = features_df.columns[column_index]
            
            
            # @var category String Get the category for the color
            category = column.split ('\n')[0]
            
            
            # @var color_index int
            color_index = categories.index (category)
            
            
            # @var color String
            color = color_palette[color_index]
            
            
            # Create histogram
            features_df.hist (
                column = column, 
                bins = bins, 
                ax = ax[i, j], 
                figsize = (width_in_inches, height_in_inches),
                color = color
            )
            column_index += 1 


    # Set title
    plt.suptitle ('Linguistic statistics for dataset %s and corpus %s' % (args.dataset, args.corpus), 
        ha = 'center', 
        fontsize = 'xx-large'
    )


    # Save figure
    fig.savefig (dataset.get_working_dir (dataset.task, 'statistics', 'lf-distribution.png'))



if __name__ == '__main__':
    main ()
    