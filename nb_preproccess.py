import os
import sys
import json
import string
import re

reload(sys)
sys.setdefaultencoding('utf-8')

def readStopWords(text_path):
    stop_words = []
    with open(text_path) as f:
        for s in f.readlines():
            word = s.lstrip()
            stop_words.append(word)
    return stop_words


def readFile(path):
    positive_reviews = {}
    negative_reviews = {}
    neutral_reviews = {}
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    texts = {}

    for dirpath, dirs, files in os.walk(path):
        for fileName in files:
            if fileName.find(".json") != -1:
                fname = os.path.join(dirpath, fileName)
                jsonfile = open(fname)
                jsondata = json.load(jsonfile)
                for product in jsondata['products']:
                    name = product['name']
                    if 'detail' in product:
                     for full_review in product['detail']:
                         star = full_review['rate'][0]
                         if 'comments' in full_review:
                             comment = full_review['comments'].lower()
                             words = comment.split()
                             if star == '1' or star == '2':
                                 negative_count += 1
                                 for word in words:
                                     if word in negative_reviews:
                                         negative_reviews [word] += 1
                                     else:
                                         negative_reviews[word] = 1
                             elif star == '4' or star == '5':
                                 positive_count += 1
                                 for word in words:
                                     if word in positive_reviews:
                                         positive_reviews[word] += 1
                                     else:
                                         positive_reviews[word] = 1
                             else:
                                 neutral_count += 1
                                 for word in words:
                                     if word in neutral_reviews:
                                         neutral_reviews[word] += 1
                                     else:
                                         neutral_reviews[word] = 1
    # print positive_reviews
    # print "\n\n\n"
    # print negative_reviews
    # print "\n\n\n"
    # print neutral_reviews

    texts['positive'] = []
    texts['positive'].append(positive_count)
    texts['positive'].append(positive_reviews)
    texts['negative'] = []
    texts['negative'].append(negative_count)
    texts['negative'].append(negative_reviews)
    texts['neutral'] = []
    texts['neutral'].append(neutral_count)
    texts['neutral'].append(neutral_reviews)
    return texts



#remove punctuations of a string and turn to lowercase
def removePunctuation(str):
    str = str.strip()  #remove /n
    str = str.translate(string.maketrans("",""), string.punctuation)
    str = str.lower()
    return str

# get frequency for each word in all classes
def getFrequency(allWords, wordDict, total):
    frequency = {}
    for i in range(0, len(allWords)):
        word = allWords[i]
        frequency[word] = float(wordDict[word]) / float(total)
    return frequency


# add-one smoothing
def smoothing(wordDict, allWords):
    result = []
    total_class = 0
    for key in wordDict.keys():
        total_class += wordDict[key]
    for i in range(0, len(allWords)):
        word = allWords[i]
        if word not in wordDict.keys():
            wordDict[word] = 1
        else:
            wordDict[word] += 1
    # update frequency
    total_class += len(wordDict)
    result.append(total_class)
    result.append(wordDict)
    return result

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


# remove short and long words
def removeShortLongwords(wordDict):
    for key in wordDict.keys():
        if len(key) <= 4 or len(key) >= 15:
            del wordDict[key]


# python preproccess.py /Users/chiling/Desktop/544/Project/Data/
if __name__ == "__main__":
    texts = readFile(sys.argv[1])
    stop_words = readStopWords("./stop_words.txt")

    #preproccess dictionary of each class
    #positive reviews
    positive = texts['positive']
    positive_count = positive[0]
    positive_dic = positive[1]
    removeStopWords(stop_words, positive_dic)
    removeDigits(positive_dic)
    removeShortLongwords(positive_dic)

    #negative reviews
    negative = texts['negative']
    negative_count = negative[0]
    negative_dic = negative[1]
    removeStopWords(stop_words, negative_dic)
    removeDigits(negative_dic)
    removeShortLongwords(negative_dic)

    #neutral reviews
    neutral = texts['neutral']
    neutral_count = neutral[0]
    neutral_dic = neutral[1]
    removeStopWords(stop_words, neutral_dic)
    removeDigits(neutral_dic)
    removeShortLongwords(neutral_dic)

    # calcualte prior of each class
    total = positive_count + negative_count + neutral_count
    prior_positive = float(positive_count) / float(total)
    prior_negative = float(negative_count) / float(total)
    prior_neutral = float(neutral_count) / float(total)

    # file out
    with open('nbmodel.txt', 'w') as fo:
        separator = '###'
        fo.write(separator + 'positive' + separator + str(prior_positive) + '\n')
        for key in positive_dic.keys():
            fo.write(key + separator + str(positive_dic[key]) + '\n')

        fo.write(separator + 'negative' + separator + str(prior_negative) + '\n')
        for key in negative_dic.keys():
            fo.write(key + separator + str(negative_dic[key]) + '\n')

        fo.write(separator + 'neutral' + separator + str(prior_neutral) + '\n')
        for key in neutral_dic.keys():
            fo.write(key + separator + str(neutral_dic[key]) + '\n')