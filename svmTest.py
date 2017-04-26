from sklearn import svm

if __name__ == "__main__":
    X = [[0,1,1], [1,1,1]]
    y = ["negative", "positive"]
    clf = svm.SVC()
    clf.fit(X, y)
    result = clf.predict([[0,1,1]])
    print(result)