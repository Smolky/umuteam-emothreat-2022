"""
    Generate model diagram
    
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
from tensorflow.keras.utils import plot_model

def main ():

    # var parser
    parser = DefaultParser (description = '')
    
    
    # Add folder name
    parser.add_argument ('--folder', dest = 'folder', default = '', help = 'Select a name of the model')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()
    

    # @var training_resume_file String
    training_resume_file = dataset.get_working_dir (dataset.task, 'models', args.folder, 'training_resume.json')

    
    # Normal behaviour. The resume file exists
    with open (training_resume_file) as json_file:
        training_resume = json.load (json_file)
    
    
    # @var model_type String
    model_type = training_resume['model'] if 'model' in training_resume else 'deep-learning'
    
    
    # @var model Model
    model = model_resolver.get (model_type)
    model.set_folder (args.folder)
    model.set_dataset (dataset)
    model.is_merged (dataset.is_merged)
    
    
    # @var model_file_path String
    model_file_path = dataset.get_working_dir (dataset.task, 'models', model.get_folder (), 'model.png')
    
    
    # @var best_model String
    best_model = model.get_best_model ()
    

    print ("training resume")
    print ("---------------")
    print (training_resume)  
    
    print ("summary")
    print ("---------------")    
    print (best_model.summary ())
    plot_model (best_model, to_file = model_file_path)


if __name__ == "__main__":
    main ()
    