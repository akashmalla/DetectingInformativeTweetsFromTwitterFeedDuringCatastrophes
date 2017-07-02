__author__ = 'akashmalla'
import tweepy
import csv
from csv import writer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import difflib
import time

# Twitter API credentials below to retrieve tweets
access_key = "28956455-oyI4ke3TPOo5AbR2aV62xENhYfKckmwi6XSQBoDre"
access_secret = "yI2Xif32poPJDSPrAlIZ5RPeMo1JMGKQQlWI9rW3TyfRW"
consumer_key = "KeBNwqzQgHX9LWkuwfRokBo5w"
consumer_secret = "QyvED6FHMszHcW7p2BqfHR3gISfXMLShaUGOnb0rLwkwrFuAqn"

#This function allows to get
def recentCrisisTweets():

    # authorize twitter and initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    try:
        # initialize a dictionary to hold all the tweepy Tweets
        tweets={}

        #Prepare to write tweets to csv file
        out = open('/Users/akashmalla/Documents/COEN281/outputRecentTweets.csv', 'w')
        csv = writer(out)
        #Set CSV file headers
        csv.writerow(["Tweet ID","Text", "Created At","Geo Enabled","Hashtag Text","URLs"])

        #make initial request for most recent tweets (500 is the maximum allowed count)
        #for tweet in tweepy.Cursor(api.search,q=['flood coyote creek','#SanJoseFlood','#CoyoteCreek','#SanJose']).items(500):
        for tweet in api.search(q=['flood coyote creek'],
                               rpp=100,
                               include_entities=True,
                               lang="en",count=100):
            #Remove any retweets found as they would show up as duplicate tweets which we want to disregard
            #if tweet.text[0] !='R' and tweet.text[1] !='T':
                #print("Below is stream of data being captured: ")
                #print(tweet)
                tweets[tweet.id_str]=[]
                tweets[tweet.id_str].append(tweet.text)
                tweets[tweet.id_str].append(tweet.created_at)
                tweets[tweet.id_str].append(tweet.user.geo_enabled)
                if 'hashtags' in tweet.entities:
                    #print(tweet.entities['hashtags'])
                    hashtagText=[]
                    for hashtag in tweet.entities['hashtags']:
                        #print("hashtag found!!!!! ",hashtag['text'])
                        hashtagText.append(hashtag['text'])
                    tweets[tweet.id_str].append(hashtagText)
                if 'urls' in tweet.entities:
                    expandedURLs=[]
                    for url in tweet.entities['urls']:
                        #print("url found!!!!!",url['expanded_url'])
                        expandedURLs.append(url['expanded_url'])
                    tweets[tweet.id_str].append(expandedURLs)

                allHashTags = ''
                allURLs = ''
                if len(hashtagText)>1:
                    allHashTags = ' '.join(str(h) for h in hashtagText)

                if len(expandedURLs)>1:
                    allURLs = ' '.join(str(u) for u in expandedURLs)

                values=[tweet.id_str,tweet.text,tweet.created_at,tweet.user.geo_enabled,allHashTags,allURLs]
                row=[str(value) for value in values]
                csv.writerow(row)
        out.close()
    except tweepy.TweepError as e:
        print(e.reason)
        time.sleep(100)
    except Exception as e:
        print(e.reason)

    return tweets

def naiveBayesPrediction(train_data, train_class, test_data, test_class=[]):
    count_vect = CountVectorizer(stop_words='english',max_df=0.7)
    #print(count_vect.vocabulary_)
    #train_count holds a count of occurrences of bag of words in the training data set
    train_count = count_vect.fit_transform(train_data)
    #print(train_count)
    tf_transformer = TfidfTransformer(use_idf=True)#idf set to true means inverse document frequency is set to true.
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
        if set(['Donations and volunteering','Caution and advice','Other Useful Information','Sympathy and support']) == set(mnb_prediction):
            print("Count of predicted values: ")
            print("Donation and volunteering tweets predicted: ",list(mnb_prediction).count("Donations and volunteering"),"Caution and advice tweets predicted:",list(mnb_prediction).count("Caution and advice"),"Other useful info tweets predicted:",list(mnb_prediction).count("Other Useful Information"),"Sympathy and support tweets predicted:",list(mnb_prediction).count("Sympathy and support"))

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
    newTweets = recentCrisisTweets()
    print("Count of new tweets after removing retweets: ",len(newTweets))

    #Check and collect any duplicate tweets found to be more than 70% similar
    dupTweetIds=set()
    checkedId=set()
    for tweetId in list(newTweets):
        checkedId.add(tweetId)
        for tid in list(newTweets):
            if tweetId != tid and tid not in checkedId:
                #Below provides a percentage of match between tweets to either consider the tweet as unique or not.
                if difflib.SequenceMatcher(None,newTweets[tweetId][0],newTweets[tid][0]).ratio() > 0.7:
                    #print("Similar tweets: ",newTweets[tweetId][0],newTweets[tid][0])
                    dupTweetIds.add(tid)

    #Remove the duplicate tweets found from the tweets collected as per Twitter API.
    for tid in dupTweetIds:
        newTweets.pop(tid)

    print("count of filtered tweets (removal of tweets which are more than 70% similar): ",len(newTweets))

    #Collect all flood related tweets and add it to allTweets that will be used for training and testing sets
    albertaTweets = readTweetsFromCSV("/Users/akashmalla/Downloads/CrisisLexT26/2013_Alberta_floods/2013_Alberta_Floods-tweets_labeled.csv")
    philipinnesTweets = readTweetsFromCSV("/Users/akashmalla/Downloads/CrisisLexT26/2012_Philipinnes_floods/2012_Philipinnes_floods-tweets_labeled.csv")
    coloradoTweets = readTweetsFromCSV("/Users/akashmalla/Downloads/CrisisLexT26/2013_Colorado_floods/2013_Colorado_floods-tweets_labeled.csv")
    queenslandTweets = readTweetsFromCSV("/Users/akashmalla/Downloads/CrisisLexT26/2013_Queensland_floods/2013_Queensland_floods-tweets_labeled.csv")
    sardiniaTweets = readTweetsFromCSV("/Users/akashmalla/Downloads/CrisisLexT26/2013_Sardinia_floods/2013_Sardinia_floods-tweets_labeled.csv")
    manilaTweets = readTweetsFromCSV("/Users/akashmalla/Downloads/CrisisLexT26/2013_Manila_floods/2013_Manila_floods-tweets_labeled.csv")

    allTweets = albertaTweets+philipinnesTweets+coloradoTweets+queenslandTweets+sardiniaTweets+manilaTweets

    #Construct train and test sets from allTweets which has all flood related tweets
    train_data = [tweet[1] for tweet in allTweets]
    train_class = [tweet[4] for tweet in allTweets]
    test_data = [tweet[0] for tweet in newTweets.values()]

    noninformativeTweets = naiveBayesPrediction(train_data, train_class, test_data)

    #Now we try to classify a related and informative tweet into help or caution tweets
    print("\nBelow is prediction of caution/advice and donation/help related tweets:")

    #Below construct training and testing set for predicting categories withing informative tweets
    #We chose to predict for 4 categories as seen below.
    informativeTweets=[]
    for t in allTweets:
        if noninformativeTweets:
            if t[1] not in noninformativeTweets:
                if t[3]=="Donations and volunteering":
                    informativeTweets.append(t)
                elif t[3]=="Caution and advice":
                    informativeTweets.append(t)
                elif t[3]=="Other Useful Information":
                    informativeTweets.append(t)
                elif t[3]=="Sympathy and support":
                    informativeTweets.append(t)
        else:
            if t[3]=="Donations and volunteering":
                informativeTweets.append(t)
            elif t[3]=="Caution and advice":
                informativeTweets.append(t)
            elif t[3]=="Other Useful Information":
                informativeTweets.append(t)
            elif t[3]=="Sympathy and support":
                informativeTweets.append(t)

    #Construct train and test data set from informativeTweets found
    train_data = [tweet[1] for tweet in informativeTweets]
    train_class = [tweet[3] for tweet in informativeTweets]
    test_data = [tweet[0] for tweet in newTweets.values()]
    #Print the train and test dataset count
    print("Train set count for informative tweets: ",len(train_data),"Test set count for informative tweets: ",len(test_data))
    naiveBayesPrediction(train_data, train_class, test_data)

