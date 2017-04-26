import json
import os
import sys
from sklearn import svm

reload(sys)
sys.setdefaultencoding('utf-8')

def getDictionary(path):
    dic = []
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
                         if 'comments' in full_review:
                             comment = full_review['comments'].lower()
                             words = comment.split()
                             for word in words:
                                 if word not in dic:
                                     dic.append(word)
    return dic

def getFeaturesVector(words, dictionary):
    vector = []
    # print("dic: ")
    # print(len(dictionary))
    for i in dictionary:
        if i in words:
            vector.append(1)
        else:
            vector.append(0)
    # print(len(vector))
    return vector

def getSimples(path, dictionary):
    positive_reviews = []
    negative_reviews = []
    neutral_reviews = []
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
                                 features = getFeaturesVector(words, dictionary)
                                 negative_reviews.append(features)

                             elif star == '4' or star == '5':
                                 positive_count += 1
                                 features = getFeaturesVector(words, dictionary)
                                 positive_reviews.append(features)

                             else:
                                 neutral_count += 1
                                 features = getFeaturesVector(words, dictionary)
                                 neutral_reviews.append(features)
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

# load test case
def getTestCase(path,dictionary):
    result = {}
    cases = []
    reviews = []
    with open(path) as f:
        cur_review = ""
        for s in f.readlines():
            if s.find("Perplexity:") == -1:
                cur_review += s
            else:
                reviews.append(cur_review)
                words = cur_review.strip().split(" ")
                case = getFeaturesVector(words, dictionary)
                cases.append(case)
                cur_review = ""
    result['cases'] = cases
    result['reviews'] = reviews
    return result

# main
# example: python SVM.py /path/to/training/data/floder  /path/to/training/prediction/floder
#          python SVM.py /Users/chiling/Desktop/544/Project/Data/test/ /Users/chiling/Desktop/544/Project/Data/test/test_case.txt
#          python SVM.py /Users/chiling/Desktop/544Project/json/ /Users/chiling/Desktop/544Project/to_run/zhiling_to_run/review_out.txt
# python SVM.py /Users/chiling/Desktop/544Project/json/ /Users/chiling/Desktop/544Project/to_run/daniel_to_run/review_out.txt

if __name__ == "__main__":
    dictionary = getDictionary(sys.argv[1])

    samples = getSimples(sys.argv[1], dictionary)
    X = []
    Y = []
    for sample in samples['positive'][1]:
        X.append(sample)
        Y.append("positive")

    for sample in samples['negative'][1]:
        X.append(sample)
        Y.append("negative")

    clf = svm.SVC()
    clf.fit(X, Y)

    # read prediction cases file
    text = getTestCase(sys.argv[2],dictionary)
    reviews = text['reviews']
    result = clf.predict(text['cases'])
    # print(result)


    # file out
    with open('svm_output_daniel.txt', 'w') as fo:
        for i in range(0, len(result)):
            fo.write(result[i])
            fo.write("#####")
            fo.write(reviews[i])
            fo.write("#####")
            fo.write('\n')
