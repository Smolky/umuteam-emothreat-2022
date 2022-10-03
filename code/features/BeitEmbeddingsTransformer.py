from transformers import BeitFeatureExtractor, BeitModel
from PIL import Image
import requests
import sys
import csv
import os.path
import io
import numpy as np
import pandas as pd
import transformers
import torch
from pathlib import Path

from tqdm import tqdm

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator 



class BeitEmbeddingsTransformer (BaseEstimator, TransformerMixin):
    """
    Generate BERT sentence vectors

    @see config.py

    @link https://huggingface.co/docs/transformers/model_doc/beit
    """
    
    def __init__ (self, model_path = 'microsoft/beit-base-patch16-224-pt22k-ft22k', image_path = 'images', cache_file = '', field = 'tweet_id'):
        """
        @param model_path String Path or keyname of the model
        @param tokenizer_path String Path or keyname of the tokenizer
        @param field String
        @param cache_file String
        """
        super ().__init__()
        
        self.suggested_field = 'tweet_id'
        self.model_path = model_path
        self.image_path = image_path
        self.field = field or self.suggested_field
        self.cache_file = cache_file
        self.number_of_features = 768
        self.prefix = 'bi'


    def transform (self, X, **transform_params):
    
        # Return vectors from cache
        if self.cache_file and os.path.exists (self.cache_file):
            return pd.read_csv (self.cache_file, header = 0, sep = ',')
        
        
        # @var feature_extractor
        feature_extractor = BeitFeatureExtractor.from_pretrained (self.model_path)
        
        
        # @var model
        model = BeitModel.from_pretrained (self.model_path)
        
        
        def get_beit_embeddings (df):
            """
            @param df DataFrame
            """
            
            # @var cls_tokens List
            cls_tokens = []
            
            
            # Iterate over rows
            for index, row in df.iterrows ():
            
                # @var image_path String
                image_path = os.path.join (self.image_path, row[self.field])
                
                
                # @var image Image
                image = Image.open (image_path)
                
                
                # @var cls_token List
                cls_token = [0 for item in range (self.number_of_features)]
                
                try:
                    with torch.no_grad ():
                        
                        # @var inputs
                        inputs = feature_extractor (images = image, return_tensors = 'pt')
                        
                        
                        # @var outputs
                        outputs = model (**inputs)
                        
                        
                        # @var last_hidden_states
                        last_hidden_states = outputs.last_hidden_state
                        
                        
                        # @var cls_token
                        cls_token = last_hidden_states[0][0].detach ().numpy ()
                
                except:
                    pass
                
                    
                cls_tokens.append (cls_token)
                
                
            return pd.DataFrame (cls_tokens)
                
        
        # @var frames List of DataFrames
        frames = []
        
        
        # Iterate on batches
        for chunk in tqdm (np.array_split (X, min ([1000, len (X)]))):
            frames.append (get_beit_embeddings (chunk))
        
        print (frames[0])
        
        # @var features DataFrame Concat frames in row axis
        features = pd.concat (frames)
        
        
        # Assign column names
        features.columns = self.get_feature_names ()
        

        # Store
        if self.cache_file:
            features.to_csv (self.cache_file, index = False)
        
        
        # Return vectors
        return features
        
        
    def fit (self, X, y = None, **fit_params):
        return self
        
    def get_feature_names (self):
        return [self.prefix + '_' + str (x) for x in range (1, self.number_of_features + 1)]