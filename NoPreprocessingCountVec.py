import csv
import itertools
import re
import time
import os
import sys

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from sklearn import (
    linear_model, model_selection, naive_bayes, preprocessing, svm)
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,

                              RandomForestClassifier)
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
from sklearn.tree import DecisionTreeClassifier
from textblob import TextBlob
from wordcloud import WordCloud
file_name = os.path.basename(sys.argv[0])
stop_words = set(stopwords.words('english'))
stopwordsk = stopwords.words('english')


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
    tweets = BeautifulSoup(tweets, features='lxml').get_text()
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


def correction(tweets):
    textblob = TextBlob(tweets)
    tweets = textblob.correct()
    return tweets


'''
def preprocessdata(tweets):
    tweets = tweets.lower()
    # Converting to URL.
    tweets = re.sub(
        '((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^/s]+))', '', tweets)
    # Removing repeating letters; more than 2.
    tweets = re.compile(r'(.)\1{2,}', re.IGNORECASE).sub(r'\1', tweets)
    tweets = re.sub(r'[^a-zA-Z\s]', '', tweets, re.I | re.A)
    # Remove Username.
    tweets = re.sub('@[^\s]+', '', tweets)
    # Remove Punctuation.
    tweets = re.sub('[^\w\s]', "", tweets)
    # remove '#' sign.
    tweets = re.sub(r'#([^\s]+)', r'\1', tweets)
    # Make Multiple spaces into a single space.
    tweets = re.sub('[\s]+', ' ', tweets)
    tweets = re.sub('<.*?>', " ", tweets)
    # remove '&' tags.
    tweets = re.sub('&[\s]+', ' ', tweets)
    tweets = tweets.strip()
    return tweets
'''
# Plotting the Roc Curves.


def plotroccurves(modelname, y_pred_prob, fpr, tpr, roc_auc, predict):
    plt.figure()
    plt.plot(fpr, tpr, color='green',
             label='ROC curve (area = %0.2f)' % roc_auc)
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

# Plot learning curves.


def plot_learning_curve(estimator, modelname, title, X, y, ylim=None):
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    # plt.savefig(f'{file_name} {modelname}learningcurves.jpg')
    # plt.show()

# Snowball stemming the sentences, without stemming stop words.


def stem_sentences(sentence):
    stemmer = SnowballStemmer('english', ignore_stopwords=True)
    tokens = sentence.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

# TF-IDF the tweets to train the model.
# Copying the tfidf vector to transform an input string.


def featureExtraction(tweets, stopwordskl):
    # Using unigrams, bigrams, trigrams and four-grams as a feature.
    # Max DF as 0.05
    tfidfvector = TfidfVectorizer(
        lowercase=False, max_df=0.05, strip_accents='ascii', ngram_range=(1, 3))
    # countvector = CountVectorizer(ngram_range=(1, 5))
    featurestrain = tfidfvector.fit_transform(tweets)
    featureExtractionVector = tfidfvector
    return featurestrain, featureExtractionVector

# Transforming the Query entered to give the sentiment.


def featureExtractionTest(tweets, featureExtractionVector):
    featurestest = featureExtractionVector.transform(tweets)
    return featurestest


# Actual model fitting and classification.
def modelselection(X, y, modelname, featureExtractionVector):
    dictionarylist = {}
    accuracyscores = []
    precisionscores = []
    recallscores = []
    TP = []
    FP = []
    TN = []
    FN = []
    f1scores = []
    for modelname in modelname:
        if modelname == 'Naive Bayes':
            model = naive_bayes.MultinomialNB()
        if modelname == 'Logistic Regression':
            model = LogisticRegression(C=1., solver = 'lbfgs')
        if modelname == 'SVM':
            model = svm.LinearSVC()
        if modelname == 'K-NN':
            model = KNeighborsClassifier()
        if modelname == 'AdaBoost':
            model = AdaBoostClassifier()
        if modelname == 'Random Forest Classifier':
            model = RandomForestClassifier(n_estimators=100)
        if modelname == 'Gradient Boosting Classifier':
            model = GradientBoostingClassifier()

        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            predict = model.predict(X_test)
            accuracyscore = accuracy_score(y_test, predict)
            accuracyscores.append(accuracyscore)
            precisionscores.append(precision_score(y_test, predict))
            recallscores.append(recall_score(y_test, predict))
            TP.append(confusion_matrix(y_test, predict)[1][1])
            FP.append(confusion_matrix(y_test, predict)[1][0])
            TN.append(confusion_matrix(y_test, predict)[0][0])
            FN.append(confusion_matrix(y_test, predict)[0][1])
            f1score = f1_score(y_test, predict)
            f1scores.append(f1score)
            if modelname != 'SVM':
                y_pred_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
                roc_auc = roc_auc_score(y_test, y_pred_prob)
            # plot roc curves
            plotroccurves(modelname, y_pred_prob,
                          fpr, tpr, roc_auc, predict)

        # Convert the array to list and store average
        accuracyscores = np.asarray(accuracyscores)
        TP = np.asarray(TP)
        FP = np.asarray(FP)
        TN = np.asarray(TN)
        FN = np.asarray(FN)
        f1scores = np.asarray(f1scores)
        precisionscores = np.asarray(precisionscores)
        recallscores = np.asarray(recallscores)
        print(
            f"The average accuracy score for training dataset length: {len(train_index)} for {modelname}:")
        print("%0.6f (+/- %0.6f)" %
              (accuracyscores.mean(), accuracyscores.std() * 2))
        # accuracyscores = []
        print('TP = ', int(TP.mean()))
        print('FP = ', int(FP.mean()))
        print('TN = ', int(TN.mean()))
        print('FN = ', int(FN.mean()))
        print('F1 Score =', f1scores.mean())s
        print('The precision score = ', precisionscores.mean())
        print('The recall score = ', recallscores.mean())
        dictionary = dict()
        dictionary = ({modelname: {'Accuracy score': accuracyscores.mean(), 'f1score': f1scores.mean(), 'precision': precisionscores.mean(),
                                   'recall': recallscores.mean(), 'true positive': int(TP.mean()), 'true negative': int(TN.mean()), 'false positive': int(FP.mean()), 'false negative': int(FN.mean())}})
        dictionarylist.update(dictionary)
        accuracyscores = []
        TP = []
        FP = []
        TN = []
        FN = []
        f1scores = []
        precisionscores = []
        recallscores = []
        # The code below is used to enter a query to check the sentiment. Query is a sentence.
        '''
        if modelname == 'Naive Bayes':
                model = MultinomialNB()
                model.fit(X, y)
                while(1):

                    inputtext = input('Enter a string; "exit" to exit.\n')
                    if inputtext == 'exit':
                        exit(0)
                    inputtext = preprocessdata(inputtext)
                    inputtext = stem_sentences(inputtext)
                    inputtext = [inputtext]
                    inputvector = featureExtractionTest(
                        inputtext, featureExtractionVector)
                    predict = model.predict(inputvector)
                # Converting input string to Array for the vectorizer.

                # for i in range(len(inputtext)):

                #     inputtext[i] = preprocessdata(inputtext[i])
                #     inputtext[i] = stem_sentences(inputtext[i])
                #     inputvector[i] = featureExtractionTest(
                #         inputtext[i], featureExtractionVector)
                #     predict = model.predict(inputvector[i])
                #     # inputvector.clear()

                    # print(predict)
                    if predict == 1:
                        print('The sentiment score is positive')
                    elif predict == 0:
                        print('The sentiment score is negative')
                    else:
                        print('Error: Predict is: ', predict)
        '''
    return dictionarylist


if __name__ == "__main__":
    start = time.time()
    dataset = pd.read_csv('DatasetF.csv', encoding='latin-1')
    print("The length of Dataset is: ", len(dataset))
    # Making a copy of the dataset.
    dataset = dataset.copy()
    # Making the sentiment score 0 and 1.
    dataset.sentiment = dataset.sentiment.replace(4, 1)
    # dataset.tweets = dataset.tweets.apply(preprocessdata)
    print(dataset.tweets.head(10))
    # Shuffling the dataset before fitting in the model.
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    data = np.array(dataset.tweets.values)
    y = dataset.sentiment.values
    # Getting the vector used to fit the model for TFIDF to transform the input.
    X, featureExtractionVector = featureExtraction(data, stopwordsk)
    # The models to be used for training and testing.
    modelr = ['Naive Bayes', 'Logistic Regression', 'K-NN', 'SVM']
    modelk = ['Naive Bayes', 'Logistic Regression', 'AdaBoost', 'K-NN'
              'Random Forest Classifier']
    modelz = ['Naive Bayes', 'Logistic Regression',
              'SVM', 'Random Forest Classifier']
    modeltest = ['Naive Bayes']
    dictionary = modelselection(X, y, modelr, featureExtractionVector)
    end = time.time()
    print('The time taken is :', (end-start))
    # Creating a pandas dataframe
    newdf = pd.DataFrame(data=dictionary)
    print(newdf.head())
    newdf = newdf.T
    newdf.to_excel(f'{file_name}results.xlsx', encoding='utf-8',
                   sheet_name='OpinionMining')
   