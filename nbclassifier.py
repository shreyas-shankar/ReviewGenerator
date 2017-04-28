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
    separator = '#####'
    prior_positive = 0.0
    prior_negative = 0.0
    prior_neutral = 0.0
    dict_positive = {}
    dict_negative = {}
    dict_neutral = {}
    with open('./nb_model.txt') as f:
        for s in f.readlines():
            s = s.strip()
            if s.startswith(separator, 0, 5):
                # print(s)
                str = s.split(separator)
                if(len(str) == 3):
                    cur_class = str[1]
                    cur_prior = float(str[2])
                    if cur_class == 'positive':
                        prior_positive = cur_prior
                    elif cur_class == 'negative':
                        prior_negative = cur_prior
                # elif cur_class == 'neutral':
                #     prior_neutral = cur_prior
            else:
                if s.find(separator) != -1:
                    line = s.split(separator)
                    if cur_class == 'positive':
                        dict_positive[line[0]] = line[1]
                    elif cur_class == 'negative':
                        dict_negative[line[0]] = line[1]


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
    # dict['neutral'] = []
    # dict['neutral'].append(prior_neutral)
    # dict['neutral'].append(dict_neutral)
    return dict


# load test case
def getTestCase(path):
    stop_words = readStopWords("./stop_words.txt");
    casesList = {}
    with open(path) as f:
        review = dict()
        cur_review = ""
        for s in f.readlines():
            if s.find("Perplexity:") == -1:
                cur_review += " "
                cur_review += s
            else:
                cur_review = cur_review.replace('\n',"")
                review = removePunctuation(cur_review)
                wordDict = countWord(review)
                wordDict = removeStopWords(stop_words, wordDict)
                wordDict = removeDigits(wordDict)
                wordDict = removeShortLongWords(wordDict)
                casesList[cur_review] = wordDict
                cur_review = ""
    return casesList


# get length of a word dictionary
def getLen(wordDict):
    len = 0
    for key in wordDict.keys():
        len = len + int(wordDict[key])
    return len


# merge all words from different classes
def mergeAll(dict_positive, dict_negative, wordDict):
    allWords = {}
    allWords = dict(dict_positive, **dict_negative)
    # allWords = dict(allWords, **dict_neutral)
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
def NaibeBayes(dictList, dict_positive, dict_negative,
               len_positive, len_negative,
               prior_positive, prior_negative):
    result = []
    for id in dictList.keys():
        wordDict = dictList[id]
        positive = 1.0
        negative = 1.0
        # neutral = 1.0
        total = mergeAll(dict_positive, dict_negative, wordDict)
        # calculate probability
        for key in wordDict.keys():
            if key not in dict_positive.keys():
                positive = positive * float(1) / float(len_positive + total)
            else:
                positive = positive * float(int(dict_positive[key]) + 1) / float(len_positive + total)

            if key not in dict_negative.keys():
                negative = negative * float(1) / float(len_negative + total)
            else:
                negative = negative * float(int(dict_negative[key]) + 1) / float(len_negative + total)

            # if key not in dict_neutral.keys():
            #     neutral = neutral * float(1) / float(len_neutral + total)
            # else:
            #     neutral = neutral * float(int(dict_neutral[key]) + 1) / float(len_neutral + total)

            positive = positive * 1000
            negative = negative * 1000
            # neutral = neutral * 1000

        positive = float(prior_positive) * float(positive)
        negative = float(prior_negative) * float(negative)
        # neutral = float(prior_neutral) * float(neutral)
        # maxarg = max(positive, negative, neutral)
        if positive > negative:
            result.append("positive" + "#####" + id + "#####")
        elif positive < negative:
            result.append("negative" + "#####" + id + "#####")
        else:
            if prior_positive > prior_negative:
                result.append("positive" + "#####" + id + "#####")
            else:
                result.append("negative" + "#####" + id + "#####")
    return result


# main
# example: python nbclassifier.py /path/to/text/file
#          python nbclassifier.py /Users/chiling/Desktop/544/Project/Data/test/test_case.txt
#          python nbclassifier.py /Users/chiling/Desktop/544/Project/Data/test/nb_test_case.txt
#

if __name__ == "__main__":
    # get model
    model = getModel()
    prior_positive = model['positive'][0]
    prior_negative = model['negative'][0]
    dict_positive = model['positive'][1]
    dict_negative = model['negative'][1]
    # read test file
    path = sys.argv[1]

    dictList = getTestCase(path)
    # print(dictList)
    len_positive = getLen(dict_positive)
    len_negative = getLen(dict_negative)
    result = []
    result = NaibeBayes(dictList, dict_positive, dict_negative, len_positive, len_negative, prior_positive,
                        prior_negative)
    # file out
    with open('nb_output_shreyas.txt', 'w') as fo:
        for i in range(0, len(result)):
            fo.write(result[i])
            fo.write('\n')
