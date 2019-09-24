# test ability to pull from html url

import os, re, pickle, urllib
#import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from keras.models import load_model


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

#def policy_violated(predictions):
    #return sum(predictions)>0

#def main(sample_url='http://www.seriously.com/privacy-notice/'):
def main(sample_url='https://www.snap.com/en-US/privacy/privacy-policy'):
    print('Evaluating '+ sample_url)
    # get html stuff
    seriously_policy = get_html_from_url(sample_url)
    seriously_sentences = get_sentences_from_html(seriously_policy)

    # load models
    pickle_jar = '../pickles/nn_models/'
    vectorizer = pickle.load(open(os.path.join(pickle_jar,'count_vec_for_nn.pckl'), 'rb'))
    #classifier = pickle.load(open(os.path.join(pickle_jar,'clf_dummy.pckl'), 'rb'))
    nn_model = load_model(os.path.join(pickle_jar,'nn_model_first_trial.h5'))
    policy_indices = pickle.load(open(os.path.join(pickle_jar, 'policy_index_dict.pckl'),'rb'))
    print('models loaded successfully')
    # transform & classify html
    df_oos = pd.DataFrame(seriously_sentences,columns=['text'])
    df_oos = standardize_text(df_oos, 'text')
    oos_list = df_oos['text'].tolist()
    #X_oos = count_vectorizer.transform(oos_list)
    #X_oos = vectorizer.transform(seriously_sentences)
    X_oos = vectorizer.transform(oos_list)
    y_oos_pred = nn_model.predict(X_oos.toarray())
    #print(y_oos_pred)
    practiced_policies = sum(y_oos_pred>0.5)
    for pol in policy_indices:
        if practiced_policies[policy_indices[pol]] > 0:
            print(pol)
    #pol_result = policy_violated(y_oos_pred)
    #print(pol_result)
    #return pol_result

if __name__ == '__main__':
    main()
