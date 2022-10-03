"""
    Generate tables
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import glob
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import dlsmodels.utils as models_utils
import json
import sklearn

from pathlib import Path

from sklearn import preprocessing

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.FeatureResolver import FeatureResolver
from dlsmodels.ModelResolver import ModelResolver
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def main ():

    # var parser
    parser = DefaultParser (description = 'Generate tables achieved by feature set')
    
    
    # Add the models of the ensemble
    parser.add_argument ('--models', dest = 'models', default = '', help = 'Select the models for the ensemble')
    
    
    # Add features
    parser.add_argument ('--source', 
        dest = 'source', 
        default = 'test', 
        help = 'Determines the source to evaluate', 
        choices = ['all', 'train', 'test', 'val']
    )    
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)


    # @var df Ensure if we already had the data processed
    df = dataset.get ()
    
    
    # Replace the dataset to contain only the test or val-set
    if args.source in ['train', 'val', 'test']:
        dataset.default_split = args.source
    
    
    # @var df Ensure if we already had the data processed
    df = dataset.get ()
    
    
    # @var group_results_by_user Boolean In some author attribution tasks at tweet level
    #                                    the results should be reported by the mode of the 
    #                                    predictions of users
    group_results_by_user = dataset.group_results_by_user ()
    
    
    # @var task_type String
    task_type = dataset.get_task_type ()
    
    
    # @var labels List
    labels = dataset.get_available_labels ()
    
    
    # @var y_real Series
    y_real = dataset.get_vector_labels (df)
    
    
    # @var y_preds Dict
    y_preds = {
        'tweet': df['tweet'],
        'label': y_real.tolist ()
    }
    
    
    # @var reports
    reports = {}
    
    
    def callback (feature_key, y_pred, model_metadata):
        """
        @var feature_key String
        @var y_pred List
        @var model_metadata Dict
        """
        y_preds[feature_key] = y_pred.tolist ()
    
    
        # @var report Dict
        report = classification_report (
            y_true = y_real, 
            y_pred = y_pred, 
            digits = 5,
            output_dict = True,
            target_names = labels
        )
        
        
        # Store report
        reports[feature_key] = pd.DataFrame (report).T

        reports[feature_key]['precision'] = reports[feature_key]['precision'].mul (100)
        reports[feature_key]['recall'] = reports[feature_key]['recall'].mul (100)
        reports[feature_key]['f1-score'] = reports[feature_key]['f1-score'].mul (100)

    
    # @var models_to_evaluateodels List
    if args.models:
        models_to_evaluate = args.models.split (',')
    else:
        models_to_evaluate = glob.glob (dataset.get_working_dir (dataset.task, 'models', '*/'))
    
    
    # Get the results per model
    for model_to_test in models_to_evaluate:
    
        # @var training_resume_file String
        training_resume_file = dataset.get_working_dir (dataset.task, 'models', model_to_test, 'training_resume.json')

        
        # Normal behaviour. The resume file exists
        if os.path.isfile (training_resume_file):
            with open (training_resume_file) as json_file:
                training_resume = json.load (json_file)
        else:
            raise OSError (training_resume_file + " not found.")
            sys.exit ()

        
        # @var model_type String
        model_type = training_resume['model'] if 'model' in training_resume else 'deep-learning'
        
        
        # Skip no deep-learning models
        if not 'deep-learning' == model_type:
            continue
        
        
        # @var model Model
        model = model_resolver.get (model_type)
        model.set_folder (model_to_test)
        model.set_dataset (dataset)
        model.is_merged (dataset.is_merged)
        
        
        # @var feature_combinations List
        feature_combinations = training_resume['features'] if 'features' in training_resume else {}
    

        # Load all the available features
        for features in feature_combinations:
            
            # Indicate which features we are loading
            print ("loading features...")
            
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


    # @var table Dict
    table = {}
    for index, report in reports.items ():
        table[os.path.basename (os.path.normpath (index))] = report.loc['macro avg']
 
    table = pd.DataFrame (table).T
    
    print (table)
    print (table.to_latex (float_format = '%.10f'))
    

if __name__ == "__main__":
    main ()
