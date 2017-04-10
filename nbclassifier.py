import os
import string
import sys
import re

def readStopWords(text_path):
    stop_words = []
    with open(text_path) as f:
        for s in f.readlines():
            word = s.lstrip()
            stop_words.append(word)
    return stop_words

def getModel():
    separator = '###'
    prior_positive = 0.0
    prior_negative = 0.0
    prior_neutral = 0.0
    dict_positive = {}
    dict_negative = {}
    dict_neutral = {}
    with open('nbmodel.txt') as f:
        for s in f.readlines():
            s = s.strip()
            if s.startswith(separator, 0, 3):
                str = s.split(separator)
                cur_class = str[1]
                cur_prior = float(str[2])
                if cur_class == 'positive':
                    prior_positive = cur_prior
                elif cur_class == 'negative':
                    prior_negative = cur_prior
                elif cur_class == 'neutral':
                    prior_neutral = cur_prior
            else:
                line = s.split(separator)
                if cur_class == 'positive':
                    dict_positive[line[0]] = line[1]
                elif cur_class == 'negative':
                    dict_negative[line[0]] = line[1]
                elif cur_class == 'neutral':
                    dict_neutral[line[0]] = line[1]

    # print('positive ###############')
    # print(dict_positive)
    # print('negative ###############')
    # print(dict_negatice)
    # print('neutral ###############')
    # print(dict_neutral)
    dict = {}
    dict['positive'] = []
    dict['positive'].append(prior_positive)
    dict['positive'].append(dict_positive)
    dict['negative'] = []
    dict['negative'].append(prior_negative)
    dict['negative'].append(dict_negative)
    dict['neutral'] = []
    dict['neutral'].append(prior_neutral)
    dict['neutral'].append(dict_neutral)
    return dict


# load test case
def getTestCase(path):
    stop_words = readStopWords("./stop_words.txt");
    dictList = {}
    with open(path) as f:
        review = dict()
        for s in f.readlines():
            tmp = s.split(' ', 1)
            review = removePunctuation(tmp[1])
            wordDict = countWord(review)
            wordDict = removeStopWords(stop_words, wordDict)
            wordDict = removeDigits(wordDict)
            wordDict = removeShortLongWords(wordDict)
            dictList[tmp[0]] = wordDict
    return dictList


# get length of a word dictionary
def getLen(wordDict):
    len = 0
    for key in wordDict.keys():
        len = len + int(wordDict[key])
    return len


# merge all words from different classes
def mergeAll(dict_positive, dict_negative, dict_neutral, wordDict):
    allWords = {}
    allWords = dict(dict_positive, **dict_negative)
    allWords = dict(allWords, **dict_neutral)
    return len(allWords)


# count occurence of each word in the class
def countWord(review):
    count = {}
    words = review.split(' ')
    for i in range(0, len(words)):
        if words[i] in count.keys():
            count[words[i]] += 1
        else:
            count[words[i]] = 1
    return count


# remove punctuations of a string and turn to lowercase
def removePunctuation(str):
    str = str.strip()  # remove /n
    str = str.translate(string.maketrans("", ""), string.punctuation)
    str = str.lower()
    return str


# remove stop words
def removeStopWords(stop_words, wordDict):
    for key in wordDict.keys():
        if key in stop_words:
            del wordDict[key]
    return wordDict


# remove digits
def removeDigits(wordDict):
    pattern = re.compile('\d')
    for key in wordDict.keys():
        if key.isdigit() or bool(pattern.search(key)) == True:
            del wordDict[key]
    return wordDict


# remove short and long words
def removeShortLongWords(wordDict):
    for key in wordDict.keys():
        if len(key) <= 4 or len(key) >= 15:
            del wordDict[key]
    return wordDict


# Naive Bayes
def NaibeBayes(dictList, dict_positive, dict_negative, dict_neutral,
               len_positive, len_negative, len_neutral,
               prior_positive, prior_negative, prior_neutral):
    result = []
    for id in dictList.keys():
        wordDict = dictList[id]
        positive = 1.0
        negative = 1.0
        neutral = 1.0
        total = mergeAll(dict_positive, dict_negative, dict_neutral, wordDict)
        # calculate probability
        for key in wordDict.keys():
            if key not in dict_positive.keys():
                pt = pt * float(1) / float(len_positive + total)
            else:
                pt = pt * float(int(dict_positive[key]) + 1) / float(len_positive + total)
            if key not in dict_negative.keys():
                pd = pd * float(1) / float(len_negative + total)
            else:
                pd = pd * float(int(dict_negative[key]) + 1) / float(len_negative + total)
            if key not in dict_neutral.keys():
                nt = nt * float(1) / float(len_neutral + total)
            else:
                nt = nt * float(int(dict_neutral[key]) + 1) / float(len_neutral + total)
            positive = positive * 1000
            negative = negative * 1000
            neutral = neutral * 1000
            nt = nt * 1000
        positive = float(prior_positive) * float(positive)
        negative = float(prior_negative) * float(negative)
        neutral = float(prior_neutral) * float(neutral)
        maxarg = max(positive, negative, neutral)
        if maxarg == positive:
            result.append(id + ' ' + 'positive')
        elif maxarg == negative:
            result.append(id + ' ' + 'negative')
        elif maxarg == neutral:
            result.append(id + ' ' + 'neutral' )
    return result


# main
# example: python nbclassify.py /path/to/text/file
#          python nbclassify.py /Users/chiling/Desktop/544/Project/Data

if __name__ == "__main__":
    # get model
    model = getModel()
    prior_positive = model['positive'][0]
    prior_negative = model['negative'][0]
    prior_neutral = model['neutral'][0]
    dict_positive = model['positive'][1]
    dict_negative = model['negative'][1]
    dict_neutral = model['neutral'][1]
    # read test file
    path = sys.argv[1]
    dictList = getTestCase(path)
    # print(dictList)
    len_positive = getLen(dict_positive)
    len_negative = getLen(dict_negative)
    len_neutral = getLen(dict_neutral)
    result = []
    result = NaibeBayes(dictList, dict_positive, dict_negative, dict_neutral, len_positive, len_negative, len_neutral, prior_positive,
                        prior_negative, prior_neutral)
    # file out
    with open('nboutput.txt', 'w') as fo:
        for i in range(0, len(result)):
            fo.write(result[i])
            fo.write('\n')
