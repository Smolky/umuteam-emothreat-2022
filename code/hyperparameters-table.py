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
import json

from pathlib import Path
from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from dlsmodels.ModelResolver import ModelResolver

def main ():

    # var parser
    parser = DefaultParser (description = '')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var df Ensure if we already had the data processed
    df = dataset.get ()
    
    
    # @var models List
    models = glob.glob (dataset.get_working_dir (dataset.task, 'models', '*/'))
    
    
    # @var results dict
    results = {}


    # @var columns List
    columns = ['architecture', 'shape', 'num_layers', 'first_neuron', 'dropout', 'lr', 'batch_size', 'activation']
    
    
    # @var columns_type List
    columns_type = [('type/' + column) for column in columns]

    
    # Iterate over the models
    for model in models:
        
        # @var directory String
        directory = model.split ('/')[-2]
        
        
        
        # Skip...
        if directory == '*':
            continue;


        # @var training_resume_file String
        training_resume_file = dataset.get_working_dir (dataset.task, 'models', directory, 'training_resume.json')


        # No training resume file was found
        if not os.path.isfile (training_resume_file):
            continue

        
        # Get the details of te training
        with open (training_resume_file) as json_file:
            training_resume = json.load (json_file)
            
        
        # @var model_type String
        model_type = training_resume['model'] if 'model' in training_resume else 'deep-learning'
        
        
        # Only with parameters of the same type
        if 'deep-learning' != model_type:
            continue
                
        
        # @var hyperparameters_filename String    
        hyperparameters_filename = dataset.get_working_dir (dataset.task, 'models', directory, 'hyperparameters.csv')


        if not Path (hyperparameters_filename).is_file ():
            continue
        
        print ("reading... " + hyperparameters_filename)
        
        # @var hyperparameters_df DataFrame
        hyperparameters_df = pd.read_csv (hyperparameters_filename, index_col = False)


        # Get the best
        hyperparameters_df = hyperparameters_df.loc[hyperparameters_df['best'] == 1].tail (1).reset_index ()
        
        
        # Append
        results[directory] = hyperparameters_df[columns_type]
        

    # Store
    df = pd.concat (results.values (), keys = results.keys (), ignore_index = False) \
        .rename (columns = {'type/num_layers': '\# of layers', 'first_neuron': '\# of neurons'}) \
        .drop (['type/batch_size'], axis = 1)
    df = df.sort_index ()
    df = df.droplevel (level = 1)
    
    
    print (df)
    print (df.to_latex ())


if __name__ == "__main__":
    main ()
    