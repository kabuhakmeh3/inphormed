# test ability to pull from html url

import os, re, pickle, urllib
#import numpy as np
#import pandas as pd
from bs4 import BeautifulSoup




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

def policy_violated(predictions):
    return sum(predictions)>0

def main(url='http://www.seriously.com/privacy-notice/'):
    #print('DO SOMETHING')
    # get html stuff
    seriously_policy = get_html_from_url(url)
    seriously_sentences = get_sentences_from_html(seriously_policy)

    # load models
    #pickle_jar = '../../pickles/'
    pickle_jar = '/Users/Khaldoon/Dropbox/insight/inphormed/pickles/'
    vectorizer = pickle.load(open(os.path.join(pickle_jar,'cv_dummy.pckl'), 'rb'))
    classifier = pickle.load(open(os.path.join(pickle_jar,'clf_dummy.pckl'), 'rb'))

    # transform & classify html
    X_oos = vectorizer.transform(seriously_sentences)
    y_oos_pred = classifier.predict(X_oos)
    #print(y_oos_pred)
    pol_result = policy_violated(y_oos_pred)
    #print(pol_result)
    return pol_result

if __name__ == '__main__':
    main()
