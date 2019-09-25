import warnings
warnings.filterwarnings("ignore")
import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense

# load data
pickle_jar = '../pickles/nn_models/'
policy_file = 'policy_index_dict.pckl'
policies = pickle.load(open(os.path.join(pickle_jar, policy_file),'rb'))

with open('../pickles/df_nlp_no_negative_multilabel_target.pckl', 'rb') as p:
    df = pickle.load(p)

list_corpus = df['text'].tolist()
list_labels = df['target'].tolist()

X_train, X_test, y_train, y_test = train_test_split(list_corpus, 
                                                    list_labels, 
                                                    test_size=0.2, 
                                                    random_state=40)

def get_keras_metrics(y_test, y_predicted, threshold=0.5):
    # prepare targets
    y_threshold = (y_predicted>threshold)*1
    y_test_np = np.array(y_test)
    acc_result = ((y_predicted>threshold)==y_test)
    
    # get classification counts
    tp = ((y_test_np==y_threshold) & (y_test_np==1))*1
    tn = ((y_test_np==y_threshold) & (y_threshold==0))*1
    fp = ((y_test_np!=y_threshold) & (y_threshold==1))*1
    fn = ((y_test_np!=y_threshold) & (y_test_np==1))*1
    
    # metrics
    accuracy = sum(acc_result)/len(acc_result)
    precision = sum(tp)/(sum(tp)+sum(fp))
    recall = sum(tp)/(sum(tp)+sum(fn))
    f1_score = (2*sum(tp))/((2*sum(tp))+sum(fp)+sum(fn))
    tpr = sum(tp)/(sum(tp)+sum(fn))
    fpr = sum(fp)/(sum(fp)+sum(tn))
    
    return accuracy, precision, recall, f1_score, tpr, fpr

def generate_roc_values(y_test, y_predicted):
    #thres_values = np.linspace(0,1,11)
    thres_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tpr_list = []
    fpr_list = []
    for i in thres_values:
        tmp_metrics = get_keras_metrics(y_test, y_predicted, threshold=i)
        tpr_list.append(np.nanmean(tmp_metrics[4]))
        fpr_list.append(np.nanmean(tmp_metrics[5]))
    
    return tpr_list, fpr_list

def print_metrics(metrics):
    accuracy = np.nanmean(metrics[0])
    precision = np.nanmean(metrics[1])
    recall = np.nanmean(metrics[2])
    f1_score = np.nanmean(metrics[3])
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % 
                (accuracy, precision, recall, f1_score))

def main():
    print('--- Evaluating Neural Network Model ---')
    
    print('unigram & bigram model')
    vectorizer = CountVectorizer(ngram_range=(1,2)) # (1,1), (1,2), (2,2)
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)
    
    y_train_np = np.array(y_train)

    input_dim = X_train_counts.shape[1]
    
    # train model
    nn = Sequential()
    nn.add(Dense(116*2, activation="relu", input_shape=(input_dim,)))
    nn.add(Dense(116, activation="sigmoid"))
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn.fit(X_train_counts.toarray(), y_train_np)
    
    # save model
    nn_model_path = '../pickles/nn_models/'
    #pickle.dump(open(os.path.join(nn_model_path,'bigram_vectorizer.pckl'),'wb'))
    with open(os.path.join(nn_model_path,'vectorizer_uni_bigram.pckl'), 'wb') as p:
        pickle.dump(vectorizer, p, protocol=pickle.HIGHEST_PROTOCOL)
    nn.save(os.path.join(nn_model_path,'nn_model_uni_bigrams.h5'))

    # predict/test model
    y_predicted = nn.predict(X_test_counts.toarray())

    # roc values
    tpr_list, fpr_list = generate_roc_values(y_test, y_predicted)
    print('True Positive Rates:')
    print(tpr_list)
    print('False Positive Rates:')
    print(fpr_list)

    # some sample metrics
    metrics_025 = get_keras_metrics(y_test, y_predicted, threshold=0.25)
    metrics_05 = get_keras_metrics(y_test, y_predicted, threshold=0.5)
    metrics_075 = get_keras_metrics(y_test, y_predicted, threshold=0.75)
    
    # threshold
    print('Threshold = 0.25')
    print_metrics(metrics_025)
    print('Threshold = 0.50')
    print_metrics(metrics_05)
    print('Threshold = 0.75')
    print_metrics(metrics_075)

if __name__=='__main__':
    main()
