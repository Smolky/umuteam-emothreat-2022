"""
    EMOThreat Task 2 submission
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import csv
import json
import sklearn
import itertools
import re
from zipfile import ZipFile
from collections import Counter

from pathlib import Path

from dlsdatasets.DatasetResolver import DatasetResolver
from dlsmodels.ModelResolver import ModelResolver
from features.FeatureResolver import FeatureResolver
from utils.Parser import DefaultParser


def main ():

    # @var dataset_name String
    dataset_name = 'emothreat-task-2'
    
    
    # @var corpus_name String
    corpora = ['2022']
    
    
    # @var tasks List
    tasks = ['task-1', 'task-2']


    # var parser
    parser = DefaultParser (description = 'EMOThreat Task 2')


    # Add features
    parser.add_argument ('--test', 
        dest = 'test', 
        default = 'run-01', 
        help = 'Determines the run', 
    )
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()
    
    
    # @var tests Dict
    tests = {
        'run-01': {'folder': 'deep-learning-lf'},
        'run-02': {'folder': 'ensemble-tiny-mode'},
        'run-03': {'folder': 'ensemble-tiny-mean'}
    }


    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var test Dict
    test = tests[args.test]
    
    
    # @var source String
    source = 'test'
    
    
    # @var folder String
    folder = test['folder']


    # @var test_df DataFrame
    test_df = pd.DataFrame ()


    # @var model_values Dict
    model_values = {
        '2022': {
            'task-1': {},
            'task-2': {}
        }
    }
    
    
    # @var file_task_2 String
    file_task_2 = 'CIC_Run%run%_TaskB.csv'
    
    
    # Iterate corpora
    for corpus_name in corpora:
    
        # Iterate tasks
        for task_name in tasks:

            # Specify the rest of the args
            args.dataset = dataset_name
            args.corpus = corpus_name
            args.task = task_name
            
            
            # @var dataset Dataset This is the custom dataset for evaluation purposes
            dataset = dataset_resolver.get (dataset_name, corpus_name, task_name, False)
            
            
            # Determine if we need to use the merged dataset or not
            dataset.filename = dataset.get_working_dir (task_name, 'dataset.csv')
            
            
            # @var df DataFrame
            df = dataset.get ()
            
            
            # @var feature_resolver FeatureResolver
            feature_resolver = FeatureResolver (dataset)
            
            
            # @var training_resume_file String
            training_resume_file = dataset.get_working_dir (dataset.task, 'models', folder, 'training_resume.json')
            
            
            # Load the training resume
            with open (training_resume_file) as json_file:
                training_resume = json.load (json_file)


            print ("model resume")
            print ("---------------------------------------------")
            print (training_resume)
        

            # @var model_type String
            model_type = training_resume['model'] if 'model' in training_resume else 'deep-learning'
            
            
            # @var model Model
            model = model_resolver.get (model_type)
            model.set_folder (folder)
            model.set_dataset (dataset)
            model.is_merged (dataset.is_merged)

            
            # Load specific stuff
            if model_type == 'transformers':
                model.set_pretrained_model (training_resume['pretrained_model'])
            
            if model_type == 'ensemble':
                
                # Set ensemble strategy
                model.set_ensemble_strategy (training_resume['strategy'])
                
                
                # Load models
                for ensemble_model in training_resume['models']:
                    print ("loading {model}".format (model = ensemble_model))
                    model.add_model (ensemble_model)
            
            
            # Replace the dataset to contain only the test split
            dataset.df = dataset.get_split (df, 'test')
        
            
            # Replace the dataset to contain only the test or val-set
            if source in ['train', 'val', 'test']:
                dataset.default_split = source


            # @var df Ensure if we already had the data processed
            df = dataset.get ()
            
            
            # @var feature_combinations List
            feature_combinations = training_resume['features'] if 'features' in training_resume else {}

            

            def callback (feature_key, y_pred, model_metadata):
                model_values[corpus_name][task_name] = [item for item in y_pred]


            # Load all the available features
            for feature_set, features_cache in feature_combinations.items ():
                
                # Indicate what features are loaded
                print ("\t" + features_cache)
                if not Path (features_cache).is_file ():
                    print ("skip...")
                    continue
                
                
                # Set features
                model.set_features (feature_set, feature_resolver.get (feature_set, cache_file = features_cache))
            
            
            # Predict this feature set
            model.predict (callback = callback)
            
            
            # Clear session
            model.clear_session ();
                        

        # Set the results
        test_df = pd.DataFrame (model_values['2022'])
        # test_df['Tweets'] = dataset.df['tweet'].astype (str).tolist ()
        
        print (test_df)
        sys.exit ()

        # @var file_path String
        file_path = file_task_2.replace ('%run%', args.test)
        
        
        # @var answer_path String
        answer_path = dataset.get_working_dir ('..', 'runs', file_path)
        
        
        print (test_df)
        
        # Capitalize labels
        test_df['2022'] = test_df['2022'].rename (columns = {
            'tweet': 'Tweets', 
            'target': 'S/G'
        })
        
        
        # Store
        test_df['2022'].to_csv (answer_path, sep = '\t', quoting = csv.QUOTE_NONE)


    # @var zip_file_path String
    zip_file_path = dataset.get_working_dir ('..', 'runs', 'umuteam.zip')
    
    

if __name__ == "__main__":
    main ()
