"""
    Evaluate a new text or a test dataset
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import nlpaug.augmenter.word as naw
import nlpaug.flow as nafc
from tqdm import tqdm

from utils.Parser import DefaultParser
from dlsdatasets.DatasetResolver import DatasetResolver


def main ():

    # var parser
    parser = DefaultParser (description = 'Evaluate dataset')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    
    
    # Determine if we need to use the merged dataset or not
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var df DataFrame
    df = dataset.get ()
    df = df.head (100)


    # @var aug Augmenter Strategy
    # aug = naw.SynonymAug (aug_src = 'wordnet', lang = 'spa')
    aug = naw.ContextualWordEmbsAug (model_path = 'PlanTL-GOB-ES/roberta-base-bne', aug_p = 0.1)
    tqdm.pandas ()
        

    print (df['tweet'].progress_apply (aug.augment))
    print (df['tweet'])


if __name__ == "__main__":
    main ()