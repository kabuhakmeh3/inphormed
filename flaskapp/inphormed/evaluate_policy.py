# test ability to pull from html url

import warnings
warnings.filterwarnings("ignore")
#import os, re, pickle, urllib
import os, pickle, urllib
#import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from keras.models import load_model
from keras import backend as K


def standardize_text(df, text_field):
    '''Clean-up text column to prepare for tokenization
    
    Removes unwanted characters &
    Replaces them with spaces or blanks
    --
    Input
    + pandas dataframe
    + name of text column
    
    Returns
    + pandas dataframe with cleaned column
    '''
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

def get_html_from_url(url):
    html = urllib.request.urlopen(url).read()
    return html

def get_sentences_from_html(html):
    '''
    Clean-up raw html
    return a list of sentences
    '''
    soup = BeautifulSoup(html, 'html.parser')

    # kill all script and style elements
    for script in soup(['script', 'style']):
        script.decompose()

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())

    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    sentences_all = text.split('\n')
    sentences = [sen for sen in sentences_all if len(sen)>100]

    return sentences

def get_policy_action(policy):
    '''
    Parse Policies returned by NN classifier
    Return:
    + Policy name
    + Action
    '''
    perf = '_PERFORMED'
    not_perf = '_NOT_PERFORMED'
    if policy.endswith(not_perf):
        policy_name = policy[:-len(not_perf)].replace('_',' ')
        policy_action = 'Not Performed'
    else:
        policy_name = policy[:-len(perf)].replace('_',' ')
        policy_action = 'Performed'

    return policy_name, policy_action

model_files = {
        # format
        # model : [vectorizer, neural net]
        'base' : ['count_vec_for_nn.pckl', 
                  'nn_model_first_trial.h5'],
        
        'reduced_feats' : ['vectorizer_uni_bi_min_5_max_10k.pckl', 
                           'nn_model_uni_bi_min_5_max_10k.h5'],
        
        'stop_words' : ['vectorizer_uni_bi_min_5_max_10k_stop.pckl', 
                        'nn_model_uni_bi_min_5_max_10k_stop.h5'],
        
        'tfidf' : ['vectorizer_tfidf_uni_bi_min_5_max_10k.pckl', 
                   'nn_model_tfidf_uni_bi_min_5_max_10k.h5']
        }

#def main(url='https://www.snap.com/en-US/privacy/privacy-policy', model_name='base'):
def main(url='http://www.seriously.com/privacy-notice/', model_name='reduced_feats'):
    
    # refresh session
    K.clear_session()

    # start 
    #print('-------------------------------')
    #print('Evaluating '+ model_name.upper())
    
    # get html stuff
    seriously_policy = get_html_from_url(url)
    seriously_sentences = get_sentences_from_html(seriously_policy)

    # load models
    base_path = '/Users/Khaldoon/Dropbox/insight/inphormed/'
    pickle_path = 'pickles/nn_models/'
    pickle_jar = base_path + pickle_path
    vectorizer = pickle.load(open(os.path.join(pickle_jar,model_files[model_name][0]), 'rb'))
    nn_model = load_model(os.path.join(pickle_jar,model_files[model_name][1]))
    #vectorizer = pickle.load(open(os.path.join(pickle_jar,'count_vec_for_nn.pckl'), 'rb'))
    #nn_model = load_model(os.path.join(pickle_jar,'nn_model_first_trial.h5'))
    policy_indices = pickle.load(open(os.path.join(pickle_jar, 'policy_index_dict.pckl'),'rb'))
    #print('models loaded successfully')
    
    # transform & classify html
    df_oos = pd.DataFrame(seriously_sentences,columns=['text'])
    df_oos = standardize_text(df_oos, 'text')
    oos_list = df_oos['text'].tolist()
    X_oos = vectorizer.transform(oos_list)
    y_oos_pred = nn_model.predict(X_oos.toarray())
    practiced_policies = sum(y_oos_pred>0.5)
    
    result = {}
    for pol in policy_indices:
        if practiced_policies[policy_indices[pol]] > 0:
            pol_name, pol_action = get_policy_action(pol)
            result[pol_name] = pol_action

    #pol_result = policy_violated(y_oos_pred)
    #print(result)
    return result

if __name__ == '__main__':
    main()
