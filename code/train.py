"""
    Train a dataset from specific features
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import itertools

from pathlib import Path

from tqdm import tqdm

from dlsdatasets.DatasetResolver import DatasetResolver
from dlsmodels.ModelResolver import ModelResolver
from features.FeatureResolver import FeatureResolver
from utils.Parser import DefaultParser



def main ():
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()
    
    
    # var parser
    parser = DefaultParser (description = 'Train dataset')
    parser.add_argument ('--model', 
        dest = 'model', 
        default = model_resolver.get_default_choice (), 
        help = 'Select the family or algorithms to evaluate', 
        choices = model_resolver.get_choices ()
    )
    
    
    # @var choices List of list 
    choices = FeatureResolver.get_feature_combinations (['lf', 'se', 'be', 'we', 'ne', 'cf', 'bf', 'pr', 'ng', 'it', 'bi', 'rf'])
    
    
    # Add features for the deep-learning model
    parser.add_argument ('--features', 
        dest = 'features', 
        default = 'all', 
        help = 'Select the family or features to train', 
        choices = ['all'] + ['-'.join (choice) for choice in choices]
    )
    
    
    # Add folder
    parser.add_argument ('--folder', dest = 'folder', default = '', help = 'Select a folder to store the model')


    # Add the models of the ensemble
    parser.add_argument ('--models', dest = 'models', default = '', help = 'Select the models for the ensemble')


    # Add the pretrained model
    parser.add_argument ('--pretrained-model', dest = 'pretrained_model', default = '', help = 'Select the pretrained model')


    # Add the ensemble average strategy
    parser.add_argument ('--ensemble-strategy', dest = 'ensemble_strategy', default = 'average', help = 'Select the ensemble strategy')
    
    
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
    
    
    # @var model Model
    model = model_resolver.get (args.model)
    model.set_folder (args.folder)
    model.set_dataset (dataset)
    model.set_args (args)
    model.is_merged (dataset.is_merged)
    
    
    # @var task_type String
    task_type = dataset.get_task_type ()


    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)
    
    
    # @var feature_combinations List
    feature_combinations = FeatureResolver.get_feature_combinations (model.get_available_features ()) if args.features == 'all' else [args.features.split ('-')]
    
    
    # @var models List
    if args.models:
        for ensemble_model in args.models.split (','):
            model.add_model (ensemble_model)
    
    if args.model == 'ensemble':
        model.set_ensemble_strategy (args.ensemble_strategy)
    
    if args.pretrained_model != '':
        model.set_pretrained_model (args.pretrained_model)
    
    
    # Perform the training...
    model.train (force = args.force)
    
    
    # Clear session
    model.clear_session ()
    
    

if __name__ == "__main__":
    main ()