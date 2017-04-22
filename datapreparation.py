import json
import sys  # import sys package, if not already imported
reload(sys)
sys.setdefaultencoding('utf-8')
# Read data from data.json.
# Add each word to a Hash map and keep track of the count.

# Other data preparation
# convert to small case.
negative_reviews = {}
positive_reviews = {}

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
                    if star == '1' or star == '2' or star == '3':
                        print "bad"
                        for word in words:
                            if word in negative_reviews:
                                negative_reviews[word] = negative_reviews[word] + 1
                            else:
                                negative_reviews[word] = 1
                        with open("negative_reviews.txt", "a+") as badreviewfile:
                            badreviewfile.write(comment + '\n')
                    elif star == '4' or star == '5' or star == '3':
                        for word in words:
                            if word in positive_reviews:
                                positive_reviews[word]  = positive_reviews[word] + 1
                            else:
                                positive_reviews[word] = 1
                        with open("positive_reviews.txt", "a+") as goodreviewfile:
                            goodreviewfile.write(comment + '\n')






getWordCount("json/24_CellPhones.json")
getWordCount("json/27_PCs.json")
getWordCount("json/tablets_data.json")
getWordCount("json/Accessories.json")
getWordCount("json/GameAccessories.json")
getWordCount("json/games.json")
getWordCount("json/KitchenAppliances.json")
getWordCount("json/MajorAppliances.json")
getWordCount("json/navigationsystem.json")
getWordCount("json/Networks.json")
getWordCount("json/OfficeSupplies.json")
getWordCount("json/printers.json")
getWordCount("json/TVs.json")
getWordCount("json/videogames.json")



# print positive_reviews
# print "\n\n\n"
# print negative_reviews
# print "\n\n\n"
# print neutral_reviews
