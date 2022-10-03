"""
    Evaluate folds
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import itertools
import pandas as pd

from pathlib import Path

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser



def main ():
    
    # var parser
    parser = DefaultParser (description = 'Evaluate folds')
    
    
    # Add folder
    parser.add_argument ('--folder', dest = 'folder', default = '', help = 'Select a folder to store the model')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, False)
    
    
    # Determine if we need to use the merged dataset or not
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # @var df DataFrame
    df = dataset.get ()
    
    
    # @var base_folder_name String
    base_folder_name = args.folder
    
    
    # @var dfs None|DataFrame
    dfs = []
    
    
    # Iterate per folds
    for fold in range (1, 11):
        
        # Update the label
        df['__split'] = df['__split_fold_' + str (fold)]
        
        
        # @var folder_name String
        folder_name = base_folder_name + '/' + 'fold_' + str (fold)
        
        
        # @var temp DataFrame
        dfs.append (pd.read_csv (dataset.get_working_dir (args.task, 'models', folder_name, 'hyperparameters.csv')))
        
        
        # Store the fold and the run id
        dfs[-1]['custom-index'] = dfs[-1].index
        dfs[-1] = dfs[-1].assign (fold = fold)
        
            
        
    # @var df_hyperparameters DataFrame Merge all results
    df_hyperparameters = pd.concat (dfs, axis = 0, ignore_index = True)

    
    # @var df_max_best DataFrame Count how many times this combination reached the best result
    df_max_best = df_hyperparameters.groupby ('custom-index')['best'].apply (lambda x: (x == True).sum ()).reset_index (name = 'count')
    
    
    # Print the mean of each fold
    print (df_hyperparameters)
    print (df_hyperparameters.groupby ('custom-index').agg ({'objective': ['mean']}))
    print (df_max_best)
    
    
    # @var max_id int Determine the index of the hyperparameters which won
    max_id = df_max_best['count'].idxmax ()
    print (df_hyperparameters.loc[df_hyperparameters['custom-index'] == max_id])

if __name__ == "__main__":
    main ()