# check ability to import and successfully run functions from lib

import sys
import pandas as pd

path_to_module = '../tools/'
sys.path.append(path_to_module)
import nlp_preprocessing as nlp

def main():
    
    # text cleaning
    df = pd.read_csv() # or something
    df = nlp.standardize_text(df, 'text')

    # from html

    # make sentences
