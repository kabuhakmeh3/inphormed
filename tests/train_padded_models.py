import os, re, pickle
#import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
#from sklearn.metrics import classification_report, confusion_matrix

# functions

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

def preprocess_df(df):
    target_map = {'PERFORMED':1,'NOT_PERFORMED':0}
    #df = pd.read_csv('../train/SSO.csv')
    df['target'] = df.performed.map(target_map)
    #df = df[['text','target']]
    df = df[['text','target']].copy()
    df = standardize_text(df, 'text')
    return df

## ML FUNCTIONS
def fit_vectorizer(data, vec_type='count'):
    '''Create and fit a vectorizer

    Options:
    + count -> count_vectorizer
    + tfidf -> tfidf_vectorizer

    Input:
    + data - X data to fit the model
    + vec_type - name of vectorizer to use

    Returns:
    + Document-term matrix or Tf-idf-weighted document-term matrix
    + vectorizer - fitted model
    '''
    if vec_type=='count':
        vectorizer = CountVectorizer()
    elif vec_type=='tfidf':
        vectorizer = TfidfVectorizer()
    else:
        print('Please select an appropriate option')

    emb = vectorizer.fit_transform(data)

    return emb, vectorizer

def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

def train_test_model(df, policy_name, save_path):
    df = preprocess_df(df)

    # convert to ML friendly format
    list_corpus = df['text'].tolist()
    list_labels = df['target'].tolist()

    X_train, X_test, y_train, y_test = train_test_split(list_corpus,
                                                        list_labels,
                                                        test_size=0.2,
                                                        random_state=40)
    # vectorize word counts
    X_train_counts, count_vectorizer = fit_vectorizer(X_train, vec_type='count')
    #X_train_counts, count_vectorizer = fit_vectorizer(X_train, vec_type='tfidf')
    X_test_counts = count_vectorizer.transform(X_test)

    # train & test logsitic regression model
    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                         multi_class='ovr', n_jobs=-1, random_state=40)
    clf.fit(X_train_counts, y_train)
    y_predicted = clf.predict(X_test_counts)

    # save pickles
    with open(os.path.join(save_path, 'bow_'+policy_name+'.pckl'), 'wb') as cv:
        pickle.dump(count_vectorizer, cv, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_path, 'lr_'+policy_name+'.pckl'), 'wb') as lr:
        pickle.dump(clf, lr, protocol=pickle.HIGHEST_PROTOCOL)

    # check performance
    accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted)
    #print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" %
    #            (accuracy, precision, recall, f1))
    return accuracy, precision, recall, f1


def main():

    print('Training models...')
    path_to_data = '../train/annotated_sampled'
    models_to_train = ['Contact_E_Mail_Address_1stParty','Location_1stParty',
                       'Identifier_Cookie_or_similar_Tech_3rdParty',
                       'Contact_Phone_Number_1stParty','SSO']

    for policy_file in models_to_train:
        df = pd.read_csv(os.path.join(path_to_data, policy_file+'.csv'))
        print('Training ' + policy_file)
        acc, prec, rec, f1 = train_test_model(df,
                                              policy_file,
                                              '../pickles/models/augmented')
        print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" %
                    (acc, prec, rec, f1))

if __name__ =='__main__':
    main()
