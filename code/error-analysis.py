"""
Error Analysis

@link https://neptune.ai/blog/deep-dive-into-error-analysis-and-model-debugging-in-machine-learning-and-deep-learning

@author José Antonio García-Díaz <joseantonio.garcia8@um.es>
@author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import config
import warnings
import bootstrap
import os.path
import pandas as pd
import numpy as np
import sklearn

from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import mutual_info_classif

from dlsdatasets.DatasetResolver import DatasetResolver
from dlsmodels.ModelResolver import ModelResolver
from dlsmodels.BaseModel import BaseModel
from features.FeatureResolver import FeatureResolver
from utils.Parser import DefaultParser


def main ():
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()
    
    
    # var parser
    parser = DefaultParser (description = 'Performs an error analysis')
    
    
    # Add folders
    parser.add_argument ('--folder-base', 
        dest = 'folder_baseline', 
        help = 'Select the folder of the baseline model. If not baseline model is used, the system will try to use the grouding truth'
    )
    parser.add_argument ('--folder', 
        dest = 'folder', 
        help = 'Select the folder of the model'
    )
    
    
    # Add source
    parser.add_argument ('--source', 
        dest = 'source', 
        default = 'test', 
        help = 'Determines the source to make the error analysis', 
        choices = ['all', 'train', 'test', 'val']
    )
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, args.force)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # Retrieve the baseline model
    if args.folder_baseline:
    
        # @var baseline_resume_file String
        baseline_resume_file = dataset.get_working_dir (dataset.task, 'models', args.folder_baseline, 'training_resume.json')
        
        
        # @var baseline_resume Dict
        baseline_resume = BaseModel.retrieve_training_info (baseline_resume_file)

        
        # @var baseline_model Model
        baseline_model = model_resolver.get_from_resume (baseline_resume)
        baseline_model.set_folder (args.folder)
        baseline_model.set_dataset (dataset)
        baseline_model.is_merged (dataset.is_merged)


    # @var baseline_resume_file String
    resume_file = dataset.get_working_dir (dataset.task, 'models', args.folder, 'training_resume.json')


    # @var resume Dict
    resume = BaseModel.retrieve_training_info (resume_file)


    # @var model Model
    model = model_resolver.get_from_resume (resume)
    model.set_folder (args.folder)
    model.set_dataset (dataset)
    model.is_merged (dataset.is_merged)
    
    
    # @var task_type String
    task_type = dataset.get_task_type ()
    
    
    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)


    # @var df Dataframe Get the original dataframe
    df = dataset.get ()


    # @var labels List
    labels = dataset.get_available_labels ()
    
    
    # Replace the dataset to contain only the test or val-set
    if args.source in ['train', 'val', 'test']:
        dataset.default_split = args.source
        

    # @var df Dataframe Get the dataframe from the correct split
    df = dataset.get ()
    
    
    # @var y_real Series Get  real labels
    y_real = dataset.get_vector_labels (df)
    
    
    # @var y_real_labels_available boolean
    y_real_labels_available = not pd.isnull (y_real).all ()
    if 'regression' == task_type:
        y_real_labels_available = None

    
    # @var predictions_df DataFrame
    global predictions_df
    predictions_df = pd.DataFrame.from_records ({
        'tweet': df['tweet'],
        'y_real': y_real.tolist ()
    })
    
    
    def callback (feature_key, y_pred, model_metadata):
        """
        @param feature_key String
        @param y_pred List
        @param model_metadata dict
        """

        global predictions_df


        # Adapt data to multi-label tasks
        if 'multi_label' == task_type:
            y_pred = y_pred.astype(int)



        # Update real labels
        predictions_df['y_pred'] = y_pred.tolist ()

        
        # Update probabilities
        if 'probabilities' in model_metadata and 'classification' == task_type:
            
            # @var probabilities_df DataFrame
            probabilities_df = pd.DataFrame (
                model_metadata['probabilities'], 
                columns = ['probab_model_' + label for label in labels],
                index = df.index
            )
            
            
            # Attach probabilities to the dataframe
            predictions_df = pd.concat ([predictions_df, probabilities_df], axis = 1)
    
    
    # @var feature_combinations List
    feature_combinations = resume['features'] if 'features' in resume else {}


    # Load all the available features
    for feature_set, features_cache in feature_combinations.items ():
        
        # Indicate what features are loaded
        if not Path (features_cache).is_file ():
            warnings.warn ("Feature file not found: " + features_cache)
            continue
        
        
        # Set features
        model.set_features (feature_set, feature_resolver.get (feature_set, cache_file = features_cache))
            
    
    # Predict this feature set
    model.predict (callback = callback)
    model.clear_session ()

    

    # Store in the baseline the real predictions
    if not args.folder_baseline:
        predictions_df['baseline'] = y_real.tolist ()
    
    
    # Retrieve the baseline model
    else:

        def callback_baseline (feature_key, y_pred, model_metadata):
            global predictions_df
            
            
            # Update baseline predictions
            predictions_df['baseline'] = y_pred.tolist ()
    
            # Update probabilities
            if 'probabilities' in model_metadata:
                probabilities_df = pd.DataFrame (
                    model_metadata['probabilities'], 
                    columns = ['probab_base_' + label for label in labels],
                    index = df.index
                )
                    
                predictions_df = pd.concat ([predictions_df, probabilities_df], axis = 1)
    
        # @var feature_combinations List
        feature_combinations = baseline_resume['features'] if 'features' in resume else {}


        # Load all the available features
        for feature_set, features_cache in feature_combinations.items ():
            if not Path (features_cache).is_file ():
                print ("skip...")
                continue
            
            
            # Set features
            baseline_model.set_features (feature_set, feature_resolver.get (feature_set, cache_file = features_cache))


        # Predict this feature set
        baseline_model.predict (callback = callback_baseline)

        
        # Clear session
        baseline_model.clear_session ();

    
    
    # @var df_results DataFrame If we have a baseline, we keep all instances that were correctly classified 
    #                           by the baseline and not by the model
    if args.folder_baseline:
        df_results = predictions_df.loc[(predictions_df['y_real'] != predictions_df['y_pred']) & (predictions_df['baseline'] == predictions_df['y_real'])]
    else:
        df_results = predictions_df.loc[predictions_df['y_real'] != predictions_df['y_pred']]
        
        
    # @var predictions_folder String
    predictions_folder = dataset.get_working_dir (args.task, 'error-analysis', args.folder, 'output.csv')
    

    # Multi-label errors
    if 'multi_label' == task_type:
    
        # Calculate the distances with the ground truth
        df_results['distances'] = [np.sqrt (np.sum (np.array (row['y_real']) - np.array (row['y_pred'])) ** 2).astype (int) for index, row in df_results.iterrows ()]
        
        
        print ("Multi-label distance")
        count, division = pd.np.histogram (df_results['distances'], bins = [1, 2, 3, 4])
        print (count)
        print (division)
        
        
        # Sort values
        df_results = df_results.sort_values (by = 'distances', ascending = False, ignore_index = True)
    
    
    # @var df_results_subset DataFrame
    df_results_subset = df_results[['y_real', 'y_pred', 'tweet']]
    
    
    # Report the problematic results
    print (df_results_subset.head ())
    
    
    # Store wrong predictions
    df_results_subset.to_csv (predictions_folder)
    
    
    # Report the percentage of wrong classifications, compared with the real predictions
    print ()
    print ("Stats")
    print ("-----")
    print ('The wrong classifications represents the {percentage}% of total.'.format (percentage = np.round (100 * (len (df_results) / len (df)), 2)))
    
    
    # Plot the confusion matrix
    if 'classification' == task_type:
        
        print ()
        print ("Confusion matrix")
        print ("----------------")
        print (confusion_matrix (df_results['y_real'], df_results['y_pred'], labels = labels))
    
    
    # Determine what makes different
    if 'classification' == task_type:
    
        # @var feature_file String
        feature_file = feature_resolver.get_suggested_cache_file ('lf', task_type)
        
        
        # @var features_cache String The file where the features are stored
        features_cache = dataset.get_working_dir (args.task, feature_file)
        
        
        # If the feautures are not found, get the default one
        if not Path (features_cache).is_file ():
            raise Exception ('features lf file are not avaiable: ' + features_cache)
            sys.exit ()
            
            
        # @var transformer Transformer
        transformer = feature_resolver.get ('lf', cache_file = features_cache)
        
        
        # @var features_df DataFrame
        features_df = transformer.transform ([])
        
        
        # @var linguistic_features List
        linguistic_features = features_df.columns.to_list ()
        
        
        # @var df_information_gain DataFrame
        df_information_gain = df.copy ()
        
        
        # We reassign the label to discern between the correctly classified instances
        # and the wrongly classified instances
        df_information_gain['label'] = df_information_gain['label'].apply (lambda row: row.index in df_results.index.values)
        
        
        # Keep only the features in the dataset
        features_df = features_df[features_df.index.isin (df.index)].reindex (df.index)
        
        
        # Attach label
        features_df = features_df.assign (label = df_results['y_real'])
    
    
        # @var X
        X = features_df.loc[:, features_df.columns != 'label']
        
        
        # @var mi 
        mi = mutual_info_classif (X = X, y = df_information_gain['label'].tolist ()).reshape (-1, 1)
        
        
        # @var best_features_indexes List
        best_features_indexes = pd.DataFrame (mi, 
            columns = ['Coefficient'], 
            index = linguistic_features
        )

        print ()
        print ('Linguistic features coefficients')
        print ('--------------------------------')
        print (best_features_indexes.sort_values (by = 'Coefficient', ascending = False).head (5).to_csv (float_format = '%.5f'))
    

if __name__ == "__main__":
    main ()