"""
    Evaluate a new text or a test dataset
    
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
import math

from pathlib import Path

from dlsdatasets.DatasetResolver import DatasetResolver
from dlsmodels.ModelResolver import ModelResolver
from features.FeatureResolver import FeatureResolver
from utils.Parser import DefaultParser
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

from sklearn.metrics import ConfusionMatrixDisplay

from scipy.special import softmax
from utils.PrettyPrintConfussionMatrix import PrettyPrintConfussionMatrix

import seaborn as sn
import matplotlib.pyplot as plt


def main ():

    # var parser
    parser = DefaultParser (description = 'Evaluate dataset')
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()
    
    
    # @var confussion_matrix_pretty_printer PrettyPrintConfussionMatrix
    confussion_matrix_pretty_printer = PrettyPrintConfussionMatrix ()
    
    
    # Add optional architecture to evaluate
    parser.add_argument ('--architecture', 
        dest = 'architecture', 
        default = '', 
        help = 'Determines the architecture to evaluate', 
        choices = ['', 'dense', 'cnn', 'bigru', 'gru', 'lstm', 'bilstm']
    )
    
    
    # Add features
    parser.add_argument ('--source', 
        dest = 'source', 
        default = 'test', 
        help = 'Determines the source to evaluate', 
        choices = ['all', 'train', 'test', 'val']
    )
    
    
    # Add folder name
    parser.add_argument ('--folder', dest = 'folder', default = '', help = 'Select a name of the model')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var architecture String
    architecture = args.architecture
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    
    
    # Determine if we need to use the merged dataset or not
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # @var training_resume_file String
    training_resume_file = dataset.get_working_dir (dataset.task, 'models', args.folder, 'training_resume.json')

    
    # Normal behaviour. The resume file exists
    if os.path.isfile (training_resume_file):
        
        with open (training_resume_file) as json_file:
            training_resume = json.load (json_file)

    # For keeping compatibility
    else:
        
        # @var model_folder_path String
        model_folder_path = os.path.normpath (args.folder)
        
        
        # @var model_folder_parts List
        model_folder_parts = Path (model_folder_path).parts
        
        
        # @var training_resume Dict Create the training resume from scratch
        training_resume = {
            'model': model_folder_parts[0]
        }
        
        # @var feature_resolver FeatureResolver
        feature_resolver = FeatureResolver (dataset)
        
        
        if 'deep-learning' in model_folder_parts[0]:
            training_resume['features'] = {feature_set: dataset.get_working_dir (args.task, feature_resolver.get_suggested_cache_file (feature_set)) for feature_set in model_folder_parts[1].split ('-')}
        
        elif 'transformers' in model_folder_parts[0]:
            training_resume['pretrained_model'] = model_folder_parts[1]

    

    # @var model_type String
    model_type = training_resume['model'] if 'model' in training_resume else 'deep-learning'
    
    
    # @var pretrained_model String
    pretrained_model = training_resume['pretrained_model'] if model_type == 'transformers' else ''
    
    
    # @var model Model
    model = model_resolver.get (model_type)
    model.set_folder (args.folder)
    model.set_dataset (dataset)
    model.is_merged (dataset.is_merged)
    
    if pretrained_model:
        model.set_pretrained_model (pretrained_model)

    if model_type == 'ensemble':
        model.set_ensemble_strategy (training_resume['strategy'])

        for ensemble_model in training_resume['models']:
            model.add_model (ensemble_model)
    
    
    # Define the criterion for selecting the best model
    if architecture:
        model.set_best_model_criteria ({
            'architecture': architecture
        })
    
    
    # @var task_type String
    task_type = dataset.get_task_type ()


    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)
    
    
    # @var df DataFrame
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
    
    
    # @var feature_combinations List
    feature_combinations = training_resume['features'] if 'features' in training_resume else {}
    
    
    # @var y_real List|Matrix
    y_real = dataset.get_vector_labels (df)
    
    
    # @var labels List
    labels = dataset.get_available_labels ()

    
    # @var y_real_labels_available boolean
    y_real_labels_available = not pd.isnull (y_real).all ()
    if 'regression' == task_type:
        y_real_labels_available = None

    
    def callback (feature_key, y_pred, model_metadata):
        """
        @var feature_key String
        @var y_pred List
        @var model_metadata Dict
        """
        
        # @var model_name String
        model_name = model.get_folder ()
        
        
        # Reassign y_preds by user, if necessary
        y_pred = df.assign (y_pred = y_pred) \
            .groupby (by = 'user', sort = False)['y_pred'] \
            .apply (lambda x: x.mode().iloc[0]) if group_results_by_user else y_pred
        
        
        # @var report DataFrame|None
        report = pd.DataFrame (classification_report (
            y_true = y_real, 
            y_pred = y_pred, 
            digits = 5,
            output_dict = True,
            target_names = labels
        )).T if y_real_labels_available else None
        
        
        # Adjust the report in a scale from 0 to 100
        if y_real_labels_available:
            report['precision'] = report['precision'].mul (100)
            report['recall'] = report['recall'].mul (100)
            report['f1-score'] = report['f1-score'].mul (100)
        
        
        # @var probabilities_df None Assign probabilities
        probabilities_df = None
        
        if ('probabilities' in model_metadata) and model_metadata['probabilities'] is not None and (model_type not in ['machine-learning', 'ensemble', 'transformers']) and ('classification' == task_type):

            # @var do_i_have_one_label boolean To check if the probabilities encoded has one or two labels
            do_i_have_one_label = len (model_metadata['probabilities'][0]) == 1

            
            # Attach the labels
            probabilities_df = pd.DataFrame (model_metadata['probabilities'], columns = [labels[0]] if do_i_have_one_label else labels)
            
            
            # For case of binary labels calculate the contrary possibility
            if do_i_have_one_label:
                probabilities_df[labels[1]] = 1 - model_metadata['probabilities']
            
            
            # Include the real label
            probabilities_df = probabilities_df.assign (label = y_real.reset_index (drop = True))
            
            
            # Include information regarding the feature set employed
            probabilities_df = probabilities_df.assign (features = feature_key)
            
            
            # Reorder
            probabilities_df = probabilities_df[['features'] + [col for col in probabilities_df.columns if col != 'features']]
        
        
        # @var cm confusion matrix|none
        cm = None
        
        if y_real_labels_available:

            # @var labels_in_cf List
            labels_in_cf = [label.replace ('-', ' ') for label in labels]
            labels_in_cf = [(label[:12] + '...') if len (label) > 12 else label for label in labels_in_cf]

            
            # Confusion matrix for classification problems
            if 'classification' == task_type:
                
                # @var cm Confusion Matrix with percentages
                cm = sklearn.metrics.confusion_matrix (
                    y_true = y_real, 
                    y_pred = y_pred, 
                    labels = labels, 
                    normalize = 'true'
                )
                
                
                # @var cm_raw Confusion Matrix
                cm_raw = sklearn.metrics.confusion_matrix (
                    y_true = y_real,
                    y_pred = y_pred, 
                    labels = labels
                )
            
            elif 'multi_label' == task_type:
                
                cm = sklearn.metrics.multilabel_confusion_matrix (
                    y_true = np.array (y_real).astype (np.float), 
                    y_pred = np.array (y_pred).astype (np.float)
                )
            
            
            # Create confusion matrix as image
            if 'classification' == task_type:
                plt.clf ()
                ax = plt.subplot ()
                # sn.set (font_scale = .5 if (len (labels)) <= 10 else 0.3)
                sn.set (font_scale = 1)
                heatmap = sn.heatmap (cm * 100, annot = True, fmt = '.0f', annot_kws = {'size': 'small'}, cbar = None)
                
                
                if (len (labels)) <= 10:
                    ax.xaxis.set_ticklabels (labels_in_cf, rotation = 90); 
                    ax.yaxis.set_ticklabels (labels_in_cf, rotation = 30);
                    
                    for t in ax.texts: t.set_text (t.get_text() + "%")
                else:
                    for t in ax.texts: t.set_text ("")
                
                ax.set (xlabel = 'Predicted', ylabel = 'Actual')
        
        
        # @var y_pred_path String
        y_pred_path = dataset.get_working_dir (dataset.task, 'results', args.source, model_name, architecture, 'y_pred.csv')
        
        
        # @var tables_path String
        tables_path = dataset.get_working_dir (dataset.task, 'results', args.source, model_name, architecture, 'classification_report.html')
        
        
        # @var report_path String
        report_path = dataset.get_working_dir (dataset.task, 'results', args.source, model_name, architecture, 'classification_report.latex')
        
        print ("--------------------")
        print (report_path)
        
        # @var confusion_matrix_path String
        confusion_matrix_path = dataset.get_working_dir (dataset.task, 'results', args.source, model_name, architecture, 'confusion_matrix.latex')


        # @var confusion_matrix_raw_path String
        confusion_matrix_raw_path = dataset.get_working_dir (dataset.task, 'results', args.source, model_name, architecture, 'confusion_matrix_raw.latex')


        # @var confusion_matrix_heatmap_path String
        confusion_matrix_heatmap_path = dataset.get_working_dir (dataset.task, 'results', args.source, model_name, architecture, 'confusion_matrix.pdf')
        
        
        # @var probabilities_path String
        probabilities_path = dataset.get_working_dir (dataset.task, 'results', args.source, model_name, architecture, 'probabilities.csv')
        
        
        # @var weights_path String
        weights_path = dataset.get_working_dir (dataset.task, 'results', args.source, model_name, architecture, 'weights.csv')


        # @var predictions_path String
        predictions_path = dataset.get_working_dir (dataset.task, 'results', args.source, model_name, architecture, 'predictions.csv')
        
        
        # Store weights
        if 'weights' in model_metadata:
            pd.DataFrame (model_metadata['weights'], index=[0]).to_csv (weights_path, index = False)
        
        
        print ()
        print (model_name + ": " + feature_key)
        print ("================")
        
        
        # Report
        if report is not None:
            print ()
            print ("classification report")
            print ("-----------------")
            print (report.to_markdown ())
            
            
            # Store reports in other formats
            report.to_latex (report_path, index = True, float_format = '%.10f')
            report.to_html (tables_path, float_format = '%.10f')
            pd.DataFrame (np.array (y_pred).astype (np.int), columns = labels).to_csv (y_pred_path, index = False)
            

        
        # Confusion matrix for classification
        if cm is not None and (task_type == 'classification'):
            print ()
            print ("confusion matrix")
            print ("-----------------")
            confussion_matrix_pretty_printer.print (cm, labels)
            pd.DataFrame (cm).to_latex (confusion_matrix_path, index = True)
            pd.DataFrame (cm_raw).to_latex (confusion_matrix_raw_path, index = True) 
        
            plt.savefig (confusion_matrix_heatmap_path, bbox_inches = 'tight')


        # Confusion matrix for multi label classification
        if cm is not None and (task_type == 'multi_label'):
            
            # @var int num_labels Number of labels
            num_labels = len (labels)
            
            
            # @var int plots_per_row Number of plots per row
            ncols = 5
            
            if num_labels % 3 == 0:
                ncols = 3
            elif num_labels % 2 == 0:
                ncols = 2
            
            
            # @var int nrows Number of rows
            nrows = int (math.ceil (len (labels) / ncols))
            
            
            # @link https://stackoverflow.com/questions/62722416/plot-confusion-matrix-for-multilabel-classifcation-python
            f, axes = plt.subplots (nrows, ncols, figsize = (25, len (labels)))
            axes = axes.ravel ()
            
            for i in range (len (labels)):
                disp = ConfusionMatrixDisplay (confusion_matrix (y_real[:, i], y_pred[:, i]), display_labels = [0, i])
                disp.plot (ax = axes[i], values_format = '.4g')
                disp.ax_.set_title (f'{labels[i]}')
                if not (i % ncols == 0):
                    disp.ax_.set_xlabel ('')
                    
                disp.im_.colorbar.remove ()
            
            plt.subplots_adjust (wspace = 0.10, hspace = 0.1)
            f.colorbar (disp.im_, ax = axes)
            plt.savefig (confusion_matrix_heatmap_path, bbox_inches = 'tight')
            
            print (cm)
            
        
        # Save probabilities
        if probabilities_df is not None:
            probabilities_df.to_csv (probabilities_path, index = False)


        if y_real_labels_available:
            print ()
            print ('precision_recall_fscore_support (micro)')
            print ("-----------------")

            print (precision_recall_fscore_support (
                y_true = y_real, 
                y_pred = y_pred, 
                average = 'micro', 
                labels = labels if task_type == 'classification' else None)
            )
        
        
        # @var df_split Ensure if we already had the data processed
        df_split = dataset.get ()
    
        
        # Save predictions
        if 'classification' == task_type:
            pd.DataFrame ({
                'tweet': df_split['tweet_clean'],
                'y_pred': y_pred,
                'y_real': y_real
            }).to_csv (predictions_path, index = True, quoting = csv.QUOTE_ALL)


        if 'multi_label' == task_type:
            pd.DataFrame ({
                'tweet': df_split['tweet_clean'],
                'y_pred': [','.join (list (x)) for x in lb.inverse_transform (y_pred)],
                'y_real': [','.join (list (x)) for x in lb.inverse_transform (y_real)]
            }).to_csv (predictions_path, index = True, quoting = csv.QUOTE_ALL)
        
        
        # Final metric
        if 'regression' == task_type: 
            
            report = pd.DataFrame ({
                'explained variance': explained_variance_score (y_true = y_real, y_pred = y_pred),
                'mean squared log error': mean_squared_log_error (y_true = y_real, y_pred = y_pred),
                'r2': r2_score (y_true = y_real, y_pred = y_pred),
                'mae': mean_absolute_error (y_true = y_real, y_pred = y_pred),
                'mse': mean_squared_error (y_true = y_real, y_pred = y_pred),
                'rmse': np.sqrt  (mean_squared_error (y_true = y_real, y_pred = y_pred))
            }, index=[0])
            
            print ()
            print ("regression report")
            print ("-----------------")
            print (report.to_markdown ())
            
            
            # Store reports in other formats
            report.to_latex (report_path, index = True, float_format = '%.10f')
            report.to_html (tables_path, float_format = '%.10f')
            

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


if __name__ == "__main__":
    main ()
