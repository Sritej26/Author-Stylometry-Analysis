# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:53:51 2021

@author: Sritej. N
"""
import nltk
from nltk.tokenize import sent_tokenize
import re
from operator import itemgetter
from collections import Counter
import math


def most_common(instances):
    return sorted(sorted(Counter(instances).items(), key=itemgetter(0)), key=itemgetter(1), reverse=True)

T = open('Austen.txt', encoding = 'utf-8', errors = 'ignore').read()
T2= open('charles.txt', encoding = 'utf-8', errors = 'ignore').read()
def preprocess(Te):
    text = Te
    text = text.lower()
    text = text.strip(' ')			#strip removes leading and trailing characters based on argument
    text = text.replace('\n\n','\n')
    text = text.replace('\n\n','\n')
    text = text.replace('\t',' ')
    text= re.sub(r'_','',text)
    text=re.sub(r"[^\w][ivx]+\."," ",text )
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"who's","who is",text)
    text = re.sub(r"how's","how is",text)
    text = re.sub(r"'s","",text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"ain't","am not",text)
    text = re.sub(r"\'t"," not",text)
    text= re.sub(r"chapter \d+" , '', text)
    return text
text1=preprocess(T)
sentences = sent_tokenize(text1)
counts = (len(nltk.word_tokenize(sentence)) for sentence in sentences)
avg_wordperSent=sum(counts)/float(len(sentences))

words = re.findall(r"\w+", text1)
frequencies1 = most_common(words)
percentages = [(instance, count*100 / len(words)) for instance, count in frequencies1]


text2=preprocess(T2)
words2 = re.findall(r"\w+", text2)
frequencies2 = most_common(words2)

w=words+words2
freq_dist = list(nltk.FreqDist(w).most_common(30))
features=[word for word,freq in freq_dist]

feature_freqs = {}
feature_freqs[1] = {}
feature_freqs[2] = {}

for feature in features:
    presence = words.count(feature)
    feature_freqs[1][feature] = presence / len(words)

for feature in features:
    presence = words2.count(feature)
    feature_freqs[2][feature] = presence / len(words2)


corpus_features = {}
for feature in features:
    corpus_features[feature] = {}

    # Calculate the mean of the frequencies expressed in the subcorpora
    feature_average = 0
    
    feature_average += feature_freqs[1][feature]
    feature_average += feature_freqs[2][feature]
    feature_average /= 2
    corpus_features[feature]["Mean"] = feature_average

    # Calculate the standard deviation using the basic formula for a sample
    feature_stdev = 0
    diff = feature_freqs[1][feature] - corpus_features[feature]["Mean"]
    feature_stdev += diff*diff
    diff = feature_freqs[2][feature] - corpus_features[feature]["Mean"]
    feature_stdev += diff*diff
    
    feature_stdev = math.sqrt(feature_stdev)
    corpus_features[feature]["StdDev"] = feature_stdev

feature_zscores = {}
feature_zscores[1] = {}
feature_zscores[2] = {}
for feature in features:
    feature_val = feature_freqs[1][feature]
    feature_mean = corpus_features[feature]["Mean"]
    feature_stdev = corpus_features[feature]["StdDev"]
    feature_zscores[1][feature] = ((feature_val-feature_mean) /
                                            feature_stdev)

    feature_val = feature_freqs[2][feature]
    feature_mean = corpus_features[feature]["Mean"]
    feature_stdev = corpus_features[feature]["StdDev"]
    feature_zscores[2][feature] = ((feature_val-feature_mean) /
                                            feature_stdev)



T3 = open('Austen_Test.txt', encoding = 'utf-8', errors = 'ignore').read()

text3=preprocess(T3)
words3 = re.findall(r"\w+", text3)

overall = len(words3)
testcase_freqs = {}
for feature in features:
    presence = words3.count(feature)
    testcase_freqs[feature] = presence / overall

# Calculate the test case's feature z-scores
testcase_zscores = {}
for feature in features:
    feature_val = testcase_freqs[feature]
    feature_mean = corpus_features[feature]["Mean"]
    feature_stdev = corpus_features[feature]["StdDev"]
    testcase_zscores[feature] = (feature_val - feature_mean) / feature_stdev


delta1 = 0
delta2 =0
for feature in features:
    delta1 += math.fabs((testcase_zscores[feature] -
                        feature_zscores[1][feature]))
    delta2 += math.fabs((testcase_zscores[feature] -
                        feature_zscores[2][feature]))
delta1 /= len(features)
delta2 /= len(features)




