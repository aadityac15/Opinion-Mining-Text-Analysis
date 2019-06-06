import csv
import itertools
import re
import time
import os
import sys
import numpy as np
from sklearn import (
    linear_model, model_selection, naive_bayes, preprocessing, svm)
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (KFold, ShuffleSplit, StratifiedKFold,
                                     cross_validate, learning_curve)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier

import gensim
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from gensim.models import word2vec
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from textblob import TextBlob
from wordcloud import WordCloud
file_name = os.path.basename(sys.argv[0])
stop_words = set(stopwords.words('english'))
# print(stop_words)
stopwordsk = stopwords.words('english')

# Train a word2vec model with the given parameters.


def word2Vec(tweets):
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    model = word2vec.Word2Vec(tweets, workers=num_workers, iter=3,
                              size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling)
    model.init_sims(replace=True)
    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "OpinionMiningModel"
    model.save(model_name)

    return model


def preprocessdata(tweets):
    tweets = tweets.lower()
    # ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweets).split())
    # Converting to URL.
    tweets = re.sub(
        '((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^/s]+))', '', tweets)
    # Removing repeating letters; more than 2.
    tweets = re.compile(r'(.)\1{2,}', re.IGNORECASE).sub(r'\1', tweets)
    # Remove Username.
    tweets = re.sub('@[^\s]+', ' ', tweets)

    # tweets = BeautifulSoup(tweets, features='lxml').get_text()

    # Remove Punctuation.
    tweets = re.sub('[^\w\s]', "", tweets)
    # remove '#' sign.
    tweets = re.sub(r'#([^\s]+)', r'\1', tweets)
    # Make Multiple spaces into a single space.
    tweets = re.sub('[\s]+', ' ', tweets)
    tweets = re.sub('<.*?>', " ", tweets)
    # remove '&' tags.
    tweets = re.sub('&[\s]+', ' ', tweets)
    tweets = re.sub(r'[^a-zA-Z\s]', '', tweets, re.I | re.A)

    tweets = tweets.strip()
    return tweets


'''
# def preprocessstopwordsdata(tweets):
    # Remove Punctuation.
    tweets = re.sub('[^\w\s]', "", tweets)
    tweets = re.sub(r'[^a-zA-Z\s]', '', tweets, re.I | re.A)
    tweets = tweets.lower()
    tweets = tweets.strip()
    return tweets
'''
# Snowball stemming the sentences, without stemming stop words.


def stem_sentences(sentence):
    stemmer = SnowballStemmer('english')
    tokens = sentence.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

# TF-IDF the tweets to train the model.


# Transforming the Query entered to give the sentiment.
def makeFeatureVec(tweets, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 1.
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in tweets:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    # Divide the result by the number of words to get the average
    np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(tweets, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    # Initialize a counter

    counter = 0.
    # Preallocate a 2D numpy array, for speed
    tweetsFeatureVecs = np.zeros((len(tweets), num_features), dtype="float32")
    # Loop through the reviews
    for tweets in tweets:
       # Print a status message every 1000th review
        if counter % 1000. == 0.:
            print("Tweets %d of %d" % (counter, len(tweets)))
       # Call the function (defined above) that makes average feature vectors
        tweetsFeatureVecs[int(counter)] = makeFeatureVec(tweets, model,
                                                         num_features)
       # Increment the counter
        counter = counter + 1.
    return tweetsFeatureVecs


def plotroccurves(modelname, y_pred_prob, fpr, tpr, roc_auc, predict):
    plt.figure()
    plt.plot(fpr, tpr, color='green',label='ROC curve (area = %0.2f)' % roc_auc)
    plt.grid(True)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {modelname}')
    plt.legend(loc="lower right")
    plt.savefig(f'{file_name} {modelname}.jpg')
    plt.clf()
    plt.cla()
    plt.close()


def modelselectionword2vec(trainDataVecs, testDataVecs, y_train, y_test, modelname):
    accuracyscores = []
    TP = []
    FP = []
    TN = []
    FN = []
    f1scores = []
    dictionarylist = {}
    for modelname in modelname:
        if modelname == 'Logistic Regression':
            clf = LogisticRegression(C=1.0, solver = 'lbfgs')
        if modelname == 'Naive Bayes':
            clf = naive_bayes.MultinomialNB()
        if modelname == 'Random Forest':
            clf = RandomForestClassifier(n_estimators=100)
        if modelname == 'SVM':
            clf = svm.LinearSVC()
        clf.fit(trainDataVecs, y_train)
        predict = clf.predict(testDataVecs)
        accuracyscore = accuracy_score(y_test, predict)
        precision = precision_score(y_test, predict)
        recall = recall_score(y_test, predict)
        TP = (confusion_matrix(y_test, predict)[1][1])
        FP = (confusion_matrix(y_test, predict)[1][0])
        TN = (confusion_matrix(y_test, predict)[0][0])
        FN = (confusion_matrix(y_test, predict)[0][1])
        f1score = f1_score(y_test, predict)
        print(f'The accuracy score for {modelname} is', accuracyscore)
        print(f'The F1 Score for the {modelname} is :', f1_score(
            y_test, predict))
        print(f'The confusion matrix for the {modelname} is \n', confusion_matrix(
            y_test, predict))
        print('TP =', TP)
        print('TN =', TN)
        print('FP =', FP)
        print('FN =', FN)
        print('Precision = ', precision)
        print('Recall = ', recall)
        dictionary = dict()
        dictionary = ({modelname: {'Accuracy score': accuracyscore, 'f1score': f1score, 'precision': precision,
                                   'recall': recall, 'true positive': TP, 'true negative': TN, 'false positive': FP, 'false negative': FN}})
        dictionarylist.update(dictionary)
    return dictionarylist


if __name__ == "__main__":
    start = time.time()
    num_features = 300
    dataset = pd.read_csv('DatasetF.csv', encoding='latin-1')
    print("The length of Dataset is: ", len(dataset))
    """ 
    datasetwhole = pd.read_csv('onemilliontweets.csv', header=None, encoding ='latin-1')
    datasetwhole.columns = ['sentiment', 'id', 'date', 'query', 'user', 'tweets']
    datasetwhole = datasetwhole.drop(['id','date','query','user'],axis = 1)
    datasetwhole.tweets = datasetwhole.tweets.apply(preprocessdata)
    datasetwhole.tweets = datasetwhole['tweets'].apply(lambda x: ' '.join(
        [word for word in x.split() if word not in (stopwordsk)]))
    datasetwhole['tokenizedtweets'] = datasetwhole.tweets.apply(nltk.word_tokenize)
    """
    # Making a copy of the dataset.
    dataset = dataset.copy()
    # Making the sentiment score 0 and 1.
    dataset.sentiment = dataset.sentiment.replace(4, 1)
    dataset.tweets = dataset.tweets.apply(preprocessdata)
    # dataset.tweets = dataset['tweets'].apply(lambda x: ' '.join(
    #     [word for word in x.split() if word not in (stopwordsk)]))
    dataset['tokenizedtweets'] = dataset.tweets.apply(nltk.word_tokenize)
    # Function to train the model with the given parameters:
    # model = word2Vec(dataset.tokenizedtweets)
    # print(model)
    # Loading a saved model
    model = gensim.models.Word2Vec.load('OpinionMiningModel')
    # Shuffling the dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    X = dataset.tweets.values
    y = dataset.sentiment.values
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2)
    trainDataVecs = getAvgFeatureVecs(X_train, model, num_features)
    testDataVecs = getAvgFeatureVecs(X_test, model, num_features)
    imp = Imputer(missing_values=np.nan, strategy='mean')
    trainDataVecs = imp.fit_transform(trainDataVecs)
    testDataVecs = imp.fit_transform(testDataVecs)
    trainDataVecs = trainDataVecs.reshape(len(X_train), -1)
    testDataVecs = testDataVecs.reshape(len(X_test), -1)
    models = ['Logistic Regression', 'Random Forest', 'SVM']
    dictionary = modelselectionword2vec(trainDataVecs, testDataVecs,
                                        y_train, y_test, models)
    # modelselectionword2vec(trainDataVecs, testDataVecs, y_train, y_test,
    #                       models)

    # Dictionary is saved to a csv file.
    newdf = pd.DataFrame(data=dictionary)
    print(newdf.head())
    newdf = newdf.T
    newdf.to_excel(f'{file_name}results.xlsx', encoding='utf-8',
                   sheet_name='OpinionMining')
    end = time.time()
    print('The time taken : ', (end-start))
