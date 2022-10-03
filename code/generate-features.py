"""
    Generate a feature set
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import config
import bootstrap
import os.path
import json
from features.FeatureResolver import FeatureResolver
from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser


def main ():
    
    # var parser
    parser = DefaultParser (description = 'Generate a feature set')
    
    
    # Additional params
    parser.add_argument ('--feature-set', dest = 'feature_set', default = '', help = 'Select the feature set')
    parser.add_argument ('--prefix', dest = 'prefix', default = '', help = 'Select the prefix for the features. None for auto-suggestion')
    parser.add_argument ('--field', dest = 'field', default = '', help = 'Select the tweet field. None for auto-suggestion')

    
    # For feature sets based on transformers
    parser.add_argument ('--folder', dest = 'folder', default = '', help = 'Select a name of the model')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, args.force)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # @var df Dataframe
    df = dataset.get ()
    
    
    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)
    
    
    # @var cache_file String
    cache_file =  dataset.get_working_dir (args.task, (args.prefix if args.prefix else args.feature_set) + '.csv')
    
    
    # @var transformer_interface Transformer Interface
    transformer_interface = feature_resolver.get (
        args.feature_set, 
        cache_file = cache_file
    )
    
    
    # Set temporal folder
    transformer_interface.temp_folder = dataset.get_working_dir (dataset.task)
    
    
    # Set the prefix
    if args.prefix:
        transformer_interface.prefix = args.prefix
    
    
    # For features based on Transformers, we need to load the 
    # model
    if args.folder:
    
        # @var training_resume_file String
        training_resume_file = dataset.get_working_dir (dataset.task, 'models', args.folder, 'training_resume.json')
    
    
        # Normal behaviour. The resume file exists
        if os.path.isfile (training_resume_file):
            with open (training_resume_file) as json_file:
                training_resume = json.load (json_file)    
        
        # For keeping compatibility
        else:
            print ('Can\'t find the training resume file')
            sys.exit ()
            
        # @var model_type String
        model_type = training_resume['model'] if 'model' in training_resume else 'deep-learning'
        
        
        # Check
        if model_type != 'transformers':
            print ('A transformer\'s based model is required to build the sentence embeddings')
            sys.exit ()
    
    
        # Configure transformer
        transformer_interface.model_path = training_resume['pretrained_model']
        transformer_interface.tokenizer_path = training_resume['tokenizer_model'] if 'tokenizer_model' in training_resume else transformer_interface.model_path
    
    
    # Image path for vision embeddings
    if hasattr (transformer_interface, 'image_path'):
        transformer_interface.image_path = dataset.get_working_dir ('images')
    
    
    # Deal with the tokenizer
    if transformer_interface.has_tokenizer:
    
        # @var train_df DataFrame Get training split
        train_df = dataset.get_split (df, 'train')    
    
    
        # Fit with the training split
        transformer_interface.fit (train_df)
        
        
        # @var tokenizer_file String
        tokenizer_file =  dataset.get_working_dir ((args.prefix if args.prefix else args.feature_set) + '_vectorizer.pickle')
        
        
        # Store the tokenizer in disk
        transformer_interface.save_vectorizer_on_disk (tokenizer_file)
        
        
        # @var vectorizer VectorizerTransformer
        vectorizer = transformer_interface.get_vectorizer ()        

    
    # @var feature_set DataFrame. Transform (and save in cache)
    feature_set = transformer_interface.transform (df)

    
    print ('Feature set')
    print ('-----------')
    print (feature_set)


if __name__ == '__main__':
    main ()