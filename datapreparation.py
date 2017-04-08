import json
# Read data from data.json.
# Add each word to a Hash map and keep track of the count.

# Other data preparation
# convert to small case.


def getWordCount(file):
    jsonfile = open("json/24_CellPhones.json")
    jsondata = json.load(jsonfile)
    for prod in jsondata.products:
        if details in prod:
            for fullreview in prod['details']:
                star = full_review['rate'][0]
                if 'comments' in full_review:
                    with open(bad_reviews.txt, 'w') as badreviewfile:
                        comment = full_review['comments'].lower()
                        words = comment.split()
                        if star == '1' or star == '2':
                            for word in words:
                                if word in negative_reviews:
                                    negative_reviews[word] = negative_reviews[word] + 1
                                else:
                                    negative_reviews[word] = 1
                        badreviewfile.write(comment + '\n')
                    elif star == '4' or star == '5':
                        with open(good_review.txt, 'w') as goodreviewfile:
                            for word in words:
                                if word in positive_reviews:
                                    positive_reviews[word]  = positive_reviews[word] + 1
                                else:
                                    positive_reviews[word] = 1
                            goodreviewfile.write(comment + '\n')
                    else:
                        with open(neutral_reviews)
                        for word in words:
                            if word in neutral_reviews:
                                neutral_reviews[word] = neutral_reviews[word] + 1
                            else:
                                neutral_reviews[word] = 1



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


getWordCount(json/24_CellPhones.json)
getWordCount(json/27_PCs.json)
getWordCount(tablets_data.json)



print positive_reviews
print "\n\n\n"
print negative_reviews
print "\n\n\n"
print neutral_reviews
