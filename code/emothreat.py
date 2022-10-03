"""
    EMOThreat Task 1 submission
    
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

from pathlib import Path

from dlsdatasets.DatasetResolver import DatasetResolver
from dlsmodels.ModelResolver import ModelResolver
from features.FeatureResolver import FeatureResolver
from utils.Parser import DefaultParser


def main ():

    # @var dataset_name String
    dataset_name = 'emothreat'
    
    
    # @var corpus_name String
    corpora = ['2022']
    
    
    # @var tasks List
    tasks = ['']


    # var parser
    parser = DefaultParser (description = 'EMOThreat Task 1')


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
        'run-01': {'folder': 'deep-learning-all-embeddings'},
        'run-02': {'folder': 'ensemble-mode'},
        'run-03': {'folder': 'ensemble-mean'}
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
    test_df = {}

    
    
    # @var file_task_2 String
    file_task_2 = 'UMUTeam_Run%run%_TaskA.csv'
    
    
    # @var results Dict
    results = {}
    
    
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
        
            
            # @var available_labels List All the possible labels for classification and multi-label tasks
            available_labels = dataset.get_available_labels ()
            
            
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
                
                if task_name not in results:
                    results[task_name] = {}
                
                
                # @var data Dict 
                data = {

                }
                
                
                # Set data
                for index, label in enumerate (available_labels):
                    data[label] = [int (row[index]) for row in y_pred]
                
                data['Sentences'] = dataset.df['tweet'].astype (str).tolist ()
                
                # Transform to integers
                test_df[task_name] = pd.DataFrame (data)
                
                
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
        
        
        # @var file_path String
        file_path = file_task_2.replace ('%run%', args.test)
        
        
        # @var answer_path String
        answer_path = dataset.get_working_dir ('..', 'runs', file_path)
        
        
        # Reorder columns
        test_df[''] = test_df[''][['anger', 'disgust', 'fear', 'sadness', 'surprise', 'happiness', 'neutral', 'Sentences']]
        

        # Store
        test_df[''].to_csv (answer_path, quoting = csv.QUOTE_NONE, index = False)


    

if __name__ == "__main__":
    main ()
