import sys

def count(path):
    result = {}
    positive_count = 0
    negative_count = 0
    total_count = 0
    separator = "#####"
    with open(path) as f:
        for s in f.readlines():
            total_count += 1
            str = s.split(separator)
            if str[0] == 'positive':
                positive_count += 1
            elif str[0] == 'negative':
                negative_count += 1

    result['positive'] = positive_count
    result['negative'] = negative_count
    result['total'] = total_count

    return result


# python utility.py /Users/chiling/Desktop/544Project/nb_classifier_output/nb_output_shreyas.txt
if __name__ == "__main__":
    # read test file
    path = sys.argv[1]
    result = count(path)
    print(result['total'])
    print(result['positive'])
    print(result['negative'])

    # file out
    with open('evalutaion_nb_shreyas.txt', 'w') as fo:
        fo.write("total: ")
        fo.write(str(result['total']))
        fo.write('\n')

        fo.write("positive: ")
        fo.write(str(result['positive']))
        fo.write('\n')

        fo.write("negative: ")
        fo.write(str(result['negative']))
        fo.write('\n')

