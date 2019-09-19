import os, re, pickle
#import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix

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
    target_map = {'PERFORMED':1,'NOT_PERFORMED':0, 'NOT_MENTIONED':2}
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

def train_test_model(df):
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

    # check performance
    accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" %
                (accuracy, precision, recall, f1))
    print('confusion matrix')
    print(confusion_matrix(y_test, y_predicted))
    return accuracy, precision, recall, f1


def main():
    #print('Training models...')

    path_to_data = '../train/three_class'
    data_files = os.listdir(path_to_data)

    #header_line = 'policy_name, accuracy, precision, recall, f1\n'
    #with open('../reports/multiclass_bow_model_training.csv', 'w') as f:
    #    f.write(header_line)

    for policy_file in data_files:
        df = pd.read_csv(os.path.join(path_to_data, policy_file))
        print(policy_file)
        if len(df) >= 100:
            accuracy, precision, recall, f1_score = train_test_model(df)

            acc='%.3f'%(accuracy)
            prec='%.3f'%(precision)
            rec='%.3f'%(recall)
            f1='%.3f'%(f1_score)

            #line_to_write = policy_file[:-4]+', '+acc+', '+prec+', '+rec+','+f1+'\n'
            #with open('../reports/multiclass_bow_model_training.csv', 'a') as f:
            #    f.write(line_to_write)
        else:
            print('Not enough training samples for '+policy_file[:-4])
            #line_to_write = policy_file[:-4]+'\n'
            #with open('../reports/multiclass_bow_untrained_models.txt', 'a') as u:
            #    u.write(line_to_write)


if __name__ =='__main__':
    main()
