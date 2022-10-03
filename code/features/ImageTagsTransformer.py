import pandas as pd
import pickle
import os.path
import config
from tqdm import tqdm

from pathlib import Path
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from imageai.Detection import ObjectDetection



class ImageTagsTransformer (BaseEstimator, TransformerMixin):
    """
    Obtain sentences tokenized

    """

    def __init__ (self, cache_file = '', image_path = 'images', field = ''):
        """
        @param model String (see Config)
        @param cache_file String
        @param image_path String
        @param field String
        """
        super().__init__()
        
        self.suggested_field = 'twitter_id'
        self.model = config.computer_vision['yolo']
        self.cache_file = cache_file
        self.field = field or self.suggested_field
        self.image_path = image_path
        self.columns = None
        self.temp_folder = ''
        
    
    # Return self nothing else to do here
    def fit (self, X, y = None):
        return self 
        
    def transform (self, X, **transform_params):
    
        # Return tokens from cache
        if self.cache_file and os.path.exists (self.cache_file):
        
            # @var features_df DataFrame
            features_df = pd.read_csv (self.cache_file, header = 0, sep = ',')
            return features_df

        # Settings
        detector = ObjectDetection ()
        detector.setModelTypeAsYOLOv3 ()
        detector.setModelPath (self.model)
        detector.loadModel ()
        
        
        # @var probabilities List
        probabilities = []
        
        
        # @var labels Set
        labels = set ()
        
        
        # @var minimum_percentage_probability int
        minimum_percentage_probability = 51
        
        
        # Detect
        for index, row in tqdm (X.iterrows (), total = len (X.index)):
        
            # @var image_path String
            image_path = os.path.join (self.image_path, row[self.field])
        
        
            # @var detections
            _, detections = detector.detectObjectsFromImage (
                input_image = image_path,
                output_type = 'array',
                minimum_percentage_probability = minimum_percentage_probability
            )
            
            
            # Attach probabilities
            probabilities.append ([(d['name'], d['percentage_probability']) for d in detections])
            
            
            # Attach labels
            for d in detections:
                labels.add (d['name'])
            
        
        # @var w int width
        # @var h int height
        w, h = len (labels), len (X.index)
        
        
        # @var labels list
        labels = list (labels)
        
        
        # @var matrix List
        matrix = [[0 for x in range(w)] for y in range(h)] 
        
        
        # Fill the gaps
        for row_index, probability in enumerate (probabilities):
            for entity in probability:
                column_index = labels.index (entity[0])
                matrix[row_index][column_index] = entity[1]
        
        
        # @var df_features DataFrame
        features = pd.DataFrame (matrix, columns = labels)
        
        
        # Store
        if self.cache_file:
            features.to_csv (self.cache_file, index = False)
        