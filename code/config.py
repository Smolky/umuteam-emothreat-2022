"""
    Configuration of the pathss
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os

from pathlib import Path


# @var certificate String
certificate = str (Path.home ()) + '/certificates/CA.pem'


# @var umucorpusclassifier_api_endpoint String
umucorpusclassifier_api_endpoint = 'https://collaborativehealth.inf.um.es/corpusclassifier/api/'


# @var umutextstats_api_endpoint String
umutextstats_api_endpoint = 'php /home/rafa_pepe/umutextstats/api/umutextstats.php'



# @var base_path String
base_path = Path (os.path.realpath (__file__)).parent.parent 


# @var directories Paths
directories = {
    'vision': os.path.join (base_path, 'embeddings', 'vision'),
    'datasets': os.path.join (base_path, 'datasets'),
    'pretrained': os.path.join (base_path, 'embeddings', 'pretrained'),
    'assets': os.path.join (base_path, 'assets'),
    'cache': os.path.join (base_path, 'cache_dir'),
}


# @var pretrained_models 
pretrained_models = {
    'es': {
        'fasttext': {
            'binary': os.path.join (directories['pretrained'], 'cc.es.300.bin'),
            'vectors': os.path.join (directories['pretrained'], 'cc.es.300.vec'),
        },
        
        'word2vec': {
            'vectors': os.path.join (directories['pretrained'], 'word2vec-sbwc.txt')
        },
        
        'glove': {
            'vectors': os.path.join (directories['pretrained'], 'glove-sbwc.vec')
        },
    },
    
    'en': {
        'fasttext': {
            'binary': os.path.join (directories['pretrained'], 'cc.en.300.bin'),
            'vectors': os.path.join (directories['pretrained'], 'cc.en.300.vec')
        },
        
        'glove': {
            'vectors': os.path.join (directories['pretrained'], 'glove.6b.300d.txt'),
        }
    },
    
    'hi': {
        'fasttext': {
            'binary': os.path.join (directories['pretrained'], 'cc.hi.300.bin'),
            'vectors': os.path.join (directories['pretrained'], 'cc.hi.300.vec')
        }
    },
    
    'mr': {
        'fasttext': {
            'binary': os.path.join (directories['pretrained'], 'cc.mr.300.bin'),
            'vectors': os.path.join (directories['pretrained'], 'cc.mr.300.vec')
        }
    },
    
    'ar': {
        'fasttext': {
            'binary': os.path.join (directories['pretrained'], 'cc.ar.300.bin'),
            'vectors': os.path.join (directories['pretrained'], 'cc.ar.300.vec')
        }
    },

    'tam': {
        'fasttext': {
            'binary': os.path.join (directories['pretrained'], 'cc.ta.300.bin'),
            'vectors': os.path.join (directories['pretrained'], 'cc.ta.300.vec')
        }
    },
    
    'ur': {
        'fasttext': {
            'binary': os.path.join (directories['pretrained'], 'cc.ur.300.bin'),
            'vectors': os.path.join (directories['pretrained'], 'cc.ur.300.vec')
        }
    }    
}


# @var computer_vision Dict 
computer_vision = {
    'yolo': os.path.join (directories['vision'], 'yolo.h5'),
}