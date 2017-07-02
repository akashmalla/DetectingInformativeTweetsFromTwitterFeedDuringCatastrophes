# DetectingInformativeTweetsFromTwitterFeedDuringCatastrophes

Project by Akash Malla and Samarth Mehta

The purpose of this project is to provide valid information regarding any catastrophe that occurs around the world. We do not want people to believe in fake donations or browse through spam information from social media.

This project consists of the following files:
1. Project Defence
2. Project Report
3. Sample Input to run the code
4. Source Code files (tweetPrediction.py and tweetAccuracy.py)

Follow the below instructions to run the application:
1. Ensure to update tweetPrediction.py and tweetAccuracy.py to point to the sample input csvs. See below:
    albertaTweets = readTweetsFromCSV("<provide input path here>")
    philipinnesTweets = readTweetsFromCSV(""<provide input path here>"")
    coloradoTweets = readTweetsFromCSV(""<provide input path here>"")
    queenslandTweets = readTweetsFromCSV(""<provide input path here>"")
    sardiniaTweets = readTweetsFromCSV(""<provide input path here>"")
    manilaTweets = readTweetsFromCSV(""<provide input path here>"")
2. Update the following line in tweetPrediction.py (this is to keep a copy of data from search api from being lost):
out = open('<provide output path here>', 'w')
2. Ensure python version used 3.6
3. Make sure you import the following for the 2 python files to run:
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
