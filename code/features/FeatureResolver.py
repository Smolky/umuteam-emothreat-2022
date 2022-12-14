import config

import itertools

from .SentenceEmbeddingsTransformer import SentenceEmbeddingsTransformer
from .BertEmbeddingsTransformer import BertEmbeddingsTransformer
from .LinguisticFeaturesTransformer import LinguisticFeaturesTransformer
from .TokenizerTransformer import TokenizerTransformer
from .NegationTransformer import NegationTransformer
from .ContextualFeaturesTransformer import ContextualFeaturesTransformer
from .PredictionsTransformer import PredictionsTransformer
from .NGramsTransformer import NGramsTransformer
from .ImageTagsTransformer import ImageTagsTransformer
from .BeitEmbeddingsTransformer import BeitEmbeddingsTransformer


class FeatureResolver ():
    """
    FeatureResolver
    
    Determines which feature set should I load
    """
    
    @staticmethod
    def get_feature_combinations (features):
        """
        List Get all the keys of the feature sets we are going to use
        Expand it to have features in isolation and combination (lf), (lf, se), ...
        
        @return List
        """
        
        # @var feature_combinations 
        feature_combinations = [key for key in features]
        feature_combinations = [list (subset) \
                                    for L in range (1, len (feature_combinations) + 1) \
                                        for subset in itertools.combinations (feature_combinations, L)]

        return feature_combinations

    def __init__ (self, dataset):
        """
        @param dataset DatasetBase
        """
        self.dataset = dataset
    
    
    def get_suggested_cache_file (self, features, task_type = 'classification'):
        """
        Returns the suggested cache file for each feature set. The interesting point 
        here is that each feature set could have a better feature selection technique
        
        @param features String
        @param task_type String
        
        @todo It will be interesting that the features will be part of the hyperparameter 
              tunning
        
        @return String
        """
        
        if 'lf' == features:
            return 'lf.csv' if task_type in ['multi_label', 'classification'] else 'lf_minmax_regression.csv'

        if 'se' == features:
            return 'se.csv' if task_type in ['multi_label', 'classification'] else 'se_regression.csv'

        if 'be' == features:
            return 'be.csv' if task_type in ['multi_label', 'classification'] else 'be_regression.csv'
            
        if 'bf' == features:
            return 'bf.csv'
    
        if 'we' == features:
            return 'we.csv'
            
        if 'ne' == features:
            return 'ne.csv'
            
        if 'cf' == features:
            return 'cf_robust.csv'

        if 'pr' == features:
            return 'pr.csv'

        if 'ng' == features:
            return 'ng.csv'

        if 'cg' == features:
            return 'cg_ig.csv'
            
        if 'it' == features:
            return 'it_lsa_2.csv'

        if 'bi' == features:
            return 'bi.csv'

        if 'rf' == features:
            return 'rf.csv'


    def get (self, features, cache_file):
        
        if 'lf' == features:
            return LinguisticFeaturesTransformer (cache_file = cache_file)

        if 'se' == features:
        
            # @var fasttext_model String
            fasttext_model = config.pretrained_models[self.dataset.get_dataset_language ()]['fasttext']['binary']
        
            return SentenceEmbeddingsTransformer (fasttext_model, cache_file = cache_file)

        if features in ['bf', 'rf']:
            return BertEmbeddingsTransformer ('', cache_file = cache_file)

        if 'we' == features:
            return TokenizerTransformer (cache_file = cache_file)
            
        if 'ne' == features:
            return NegationTransformer (cache_file = cache_file)

        if 'cf' == features:
            return ContextualFeaturesTransformer (cache_file = cache_file)

        if 'pr' == features:
            return PredictionsTransformer (cache_file = cache_file)
            
        if 'ng' == features:
            return NGramsTransformer (cache_file = cache_file)

        if 'it' == features:
            return ImageTagsTransformer (cache_file = cache_file)
        
        if 'bi' == features:
            return BeitEmbeddingsTransformer ('microsoft/beit-base-patch16-224-pt22k-ft22k', cache_file = cache_file)
