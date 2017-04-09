import json
import sys  # import sys package, if not already imported
reload(sys)
sys.setdefaultencoding('utf-8')
# Read data from data.json.
# Add each word to a Hash map and keep track of the count.

# Other data preparation
# convert to small case.


def getWordCount(jfile):
    jsonfile = open(jfile)
    jsondata = json.load(jsonfile)
    for prod in jsondata['products']:
        if 'detail' in prod:
            for fullreview in prod['detail']:
                star = fullreview['rate']
                star = star[0]
                print star
                if 'comments' in fullreview:
                    comment = fullreview['comments']
                    print comment
                    words = comment.split()
                    if star == '1' or star == '2':
                        print "bad"
                        for word in words:
                            if word in negative_reviews:
                                negative_reviews[word] = negative_reviews[word] + 1
                            else:
                                negative_reviews[word] = 1
                        with open("negative_reviews.txt", "a+") as badreviewfile:
                            badreviewfile.write(comment + '\n')
                    elif star == '4' or star == '5':
                        for word in words:
                            if word in positive_reviews:
                                positive_reviews[word]  = positive_reviews[word] + 1
                            else:
                                positive_reviews[word] = 1
                        with open("positive_reviews.txt", "a+") as goodreviewfile:
                            goodreviewfile.write(comment + '\n')
                    else:
                        for word in words:
                            if word in neutral_reviews:
                                neutral_reviews[word] = neutral_reviews[word] + 1
                            else:
                                neutral_reviews[word] = 1
                        with open("neutral_reviews.txt", "a+") as neutralreviewfile:
                            neutralreviewfile.write(comment + '\n')





jsonfile = open("json/run_results.json")
jsondata = json.load(jsonfile)
positive_reviews = {}
negative_reviews = {}
neutral_reviews = {}
for full_review in jsondata['data']:
    star = full_review['star'][0]
    if 'comment' in full_review:
        comment = full_review['comment'].lower()
        words = comment.split()
        if star == '1' or star == '2':
            for word in words:
                if word in negative_reviews:
                    negative_reviews[word] = negative_reviews[word] + 1
                else:
                    negative_reviews[word] = 1
        elif star == '4' or star == '5':
            for word in words:
                if word in positive_reviews:
                    positive_reviews[word]  = positive_reviews[word] + 1
                else:
                    positive_reviews[word] = 1
        else:
            for word in words:
                if word in neutral_reviews:
                    neutral_reviews[word] = neutral_reviews[word] + 1
                else:
                    neutral_reviews[word] = 1


getWordCount("json/24_CellPhones.json")
getWordCount("json/27_PCs.json")
getWordCount("json/tablets_data.json")



# print positive_reviews
# print "\n\n\n"
# print negative_reviews
# print "\n\n\n"
# print neutral_reviews
