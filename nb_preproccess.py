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
    positive_reviews = []
    negative_reviews = []
    positive_count = 0
    negative_count = 0
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
                             # words = comment.split()
                             if star == '1' or star == '2':
                                 negative_count += 1
                                 negative_reviews.append(comment)
                                 # for word in words:
                                 #     if word in negative_reviews:
                                 #         negative_reviews[word] += 1
                                 #     else:
                                 #         negative_reviews[word] = 1
                             elif star == '4' or star == '5':
                                 positive_count += 1
                                 positive_reviews.append(comment)
                                 # for word in words:
                                 #     if word in positive_reviews:
                                 #         positive_reviews[word] += 1
                                 #     else:
                                 #         positive_reviews[word] = 1
    # print positive_reviews
    # print "\n\n\n"
    # print negative_reviews
    # print "\n\n\n"

    texts['positive'] = []
    texts['positive'].append(positive_count)
    texts['positive'].append(positive_reviews)
    texts['negative'] = []
    texts['negative'].append(negative_count)
    texts['negative'].append(negative_reviews)
    return texts


# count occurence of each word in the class
def countWord(reviews):
    count = {}
    for i in range(0, len(reviews)):  # traverse list of reiews
        words = reviews[i].split(' ')  # get list of words in a single review
        for j in range(0, len(words)):  # traverse list of words
            if words[j] in count.keys():
                count[words[j]] += 1
            else:
                count[words[j]] = 1
    return count

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


# python nb_preproccess.py /Users/chiling/Desktop/544/Project/Data/test
# python nb_preproccess.py /Users/chiling/Desktop/544Project/json
if __name__ == "__main__":
    texts = readFile(sys.argv[1])
    stop_words = readStopWords("./stop_words.txt")

    #preproccess dictionary of each class
    #positive reviews
    positive = texts['positive']
    positive_count = positive[0]
    positive_review = positive[1]
    positive_dic = countWord(positive_review)
    removeStopWords(stop_words, positive_dic)
    removeDigits(positive_dic)
    removeShortLongwords(positive_dic)

    #negative reviews
    negative = texts['negative']
    negative_count = negative[0]
    negative_review = negative[1]
    negative_dic = countWord(negative_review)
    removeStopWords(stop_words, negative_dic)
    removeDigits(negative_dic)
    removeShortLongwords(negative_dic)

    # calcualte prior of each class
    total = positive_count + negative_count
    prior_positive = float(positive_count) / float(total)
    prior_negative = float(negative_count) / float(total)

    # file out
    with open('nb_model_test.txt', 'w') as fo:
        separator = '#####'
        fo.write(separator + 'positive' + separator + str(prior_positive) + '\n')
        for key in positive_dic.keys():
            fo.write(key + separator + str(positive_dic[key]) + '\n')

        fo.write(separator + 'negative' + separator + str(prior_negative) + '\n')
        for key in negative_dic.keys():
            fo.write(key + separator + str(negative_dic[key]) + '\n')
