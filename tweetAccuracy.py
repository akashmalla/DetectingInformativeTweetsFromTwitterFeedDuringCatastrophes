__author__ = 'akashmalla'
import tweepy
import csv
from csv import writer
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import difflib
import time

# Twitter API credentials below
access_key = "28956455-oyI4ke3TPOo5AbR2aV62xENhYfKckmwi6XSQBoDre"
access_secret = "yI2Xif32poPJDSPrAlIZ5RPeMo1JMGKQQlWI9rW3TyfRW"
consumer_key = "KeBNwqzQgHX9LWkuwfRokBo5w"
consumer_secret = "QyvED6FHMszHcW7p2BqfHR3gISfXMLShaUGOnb0rLwkwrFuAqn"

def createTrainTestDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	testSet = dataset.copy()
	while len(trainSet) < trainSize:
		index = random.randrange(len(testSet))
		trainSet.append(testSet.pop(index))
	return [trainSet, testSet]

def getDatasetByClass(dataset):
	partialDataset = {}
	for i in range(len(dataset)):
		row = dataset[i]
		if (row[-1] not in partialDataset):
			partialDataset[row[-1]] = []
		partialDataset[row[-1]].append(row[1])
	return partialDataset

def naiveBayesPrediction(train_data, train_class, test_data, test_class=[]):
    count_vect = CountVectorizer(stop_words='english',max_df=0.7)
    #print(count_vect.vocabulary_)
    #train_count holds a count of occurrences of all words in the training data set
    train_count = count_vect.fit_transform(train_data)
    #print(train_count)
    tf_transformer = TfidfTransformer(use_idf=True) #idf set to true means inverse document frequency is set to true.
    #train_tf holds the frequency of words by considering term frequency and IDF
    train_tf = tf_transformer.fit_transform(train_count)
    test_count= count_vect.transform(test_data)
    test_tf = tf_transformer.fit_transform(test_count)

    #Multinomial Naive Bayes
    bayes_clf = MultinomialNB().fit(train_tf, train_class)
    mnb_prediction = bayes_clf.predict(test_tf)
    #print(mnb_prediction)

    noninformativeTweets=[]
    #If test_class is not being given then this function is used for prediction, if it is given, then we do
    #below to calculate accuracy and get some statistics.
    if test_class:
        bayes_accuracy = np.mean(mnb_prediction == test_class)
        print("Bayes accuracy: ",bayes_accuracy)
        print("Classification Report: ")
        # Two classes: 'Related and informative' and 'Related - but not informative'
        print(metrics.classification_report(test_class, mnb_prediction, target_names=list(set(test_class))))
        print("Confusion Matrix:")
        print(metrics.confusion_matrix(test_class, mnb_prediction))
    else:
        for i in range(len(test_data)):
            if mnb_prediction[i] in ['Related - but not informative','Related and informative'] and mnb_prediction[i] == 'Related - but not informative':
                print("Tweet: ", test_data[i], ". Prediction: ",mnb_prediction[i])
                noninformativeTweets.append(test_data[i])
            else:
                print("Tweet: ", test_data[i], ". Prediction: ",mnb_prediction[i])
        if set(['Donations and volunteering','Caution and advice']) == set(mnb_prediction):
            print("Count of predicted values: ")
            print("Donation and volunteering tweets predicted: ",list(mnb_prediction).count("Donations and volunteering"),"Caution and advice tweets predicted:",list(mnb_prediction).count("Caution and advice"))

    return noninformativeTweets

def readTweetsFromCSV(fileLocation):
    allTweets=[]
    with open(fileLocation) as f:
        try:
            reader = csv.reader(f)
            next(reader) #Skip the header line in csv file
            for row in reader:
                #Do not load tweets that do not have a related/informative label
                if "tweets_labeled" in fileLocation:
                    if row[4] != "Not applicable" and row[4] != "Not related":
                        allTweets.append(row)
                else:
                    allTweets.append(row)
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))

    return allTweets

if __name__ == '__main__':

    #Collect all flood related tweets and add it to allTweets that will be used for training and testing sets
    albertaTweets = readTweetsFromCSV("/Users/akashmalla/Downloads/CrisisLexT26/2013_Alberta_floods/2013_Alberta_Floods-tweets_labeled.csv")
    philipinnesTweets = readTweetsFromCSV("/Users/akashmalla/Downloads/CrisisLexT26/2012_Philipinnes_floods/2012_Philipinnes_floods-tweets_labeled.csv")
    coloradoTweets = readTweetsFromCSV("/Users/akashmalla/Downloads/CrisisLexT26/2013_Colorado_floods/2013_Colorado_floods-tweets_labeled.csv")
    queenslandTweets = readTweetsFromCSV("/Users/akashmalla/Downloads/CrisisLexT26/2013_Queensland_floods/2013_Queensland_floods-tweets_labeled.csv")
    sardiniaTweets = readTweetsFromCSV("/Users/akashmalla/Downloads/CrisisLexT26/2013_Sardinia_floods/2013_Sardinia_floods-tweets_labeled.csv")
    manilaTweets = readTweetsFromCSV("/Users/akashmalla/Downloads/CrisisLexT26/2013_Manila_floods/2013_Manila_floods-tweets_labeled.csv")

    allTweets = albertaTweets+philipinnesTweets+coloradoTweets+queenslandTweets+sardiniaTweets+manilaTweets
    splitRatio=0.67
    train, test = createTrainTestDataset(allTweets, splitRatio)
    print("Split {0} tweets into training set with {1} tweets and testing set with {2} tweets".format(len(allTweets), len(train), len(test)))

    classifyTweets = getDatasetByClass(train)
    print("Number of Related and informative tweets in dataset: ", len(classifyTweets['Related and informative']), ". Number of Related - but not informative tweets are ",len(classifyTweets['Related - but not informative']))

    train_data=[tweet[1] for tweet in train]
    train_class=[tweet[4] for tweet in train]
    test_data=[tweet[1] for tweet in test]
    test_class=[tweet[4] for tweet in test]
    print(list(set(train_class)))

    noninformativeTweets = naiveBayesPrediction(train_data, train_class, test_data, test_class)
    #naiveBayesPrediction(train_data, train_class, test_data, test_class)
    #print([tweets[key][0] for key in tweets])
    #test_data= [tweets[key][0] for key in tweets]

    #Now we try to classify a related and informative tweet into help or caution tweets
    print("\nBelow we predict caution and help related tweets:")

    informativeTweets=[]
    for t in allTweets:
        if noninformativeTweets:
            if t[1] not in noninformativeTweets:
                if t[3]=="Donations and volunteering":
                    informativeTweets.append(t)
                elif t[3]=="Caution and advice":
                    informativeTweets.append(t)
        else:
            if t[3]=="Donations and volunteering":
                informativeTweets.append(t)
            elif t[3]=="Caution and advice":
                informativeTweets.append(t)

    train, test = getDatasetByClass(informativeTweets, splitRatio)
    train_data=[tweet[1] for tweet in train]
    train_class=[tweet[3] for tweet in train]
    test_data=[tweet[1] for tweet in test]
    test_class=[tweet[3] for tweet in test]

    #Print the train and test dataset count
    print("Train data set: ", len(train_data), "Test data set: ",len(test_data))
    naiveBayesPrediction(train_data, train_class, test_data, test_class)
    #naiveBayesPrediction(train_data, train_class, test_data, test_class)






