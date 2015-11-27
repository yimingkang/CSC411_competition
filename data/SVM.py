import scipy.io
import pylab
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import cross_validation

def train_SVM():
    train = scipy.io.loadmat('labeled_images.mat')

    print "Shape of tr_images is: ", train["tr_images"].shape
    (x, y, n_images) = train["tr_images"].shape

    train_img = np.reshape(np.swapaxes(train["tr_images"], 0, 2), (n_images, x * y))
    #plt.imshow(np.swapaxes(np.reshape(train_img[0], (y, x)), 0, 1), cmap=pylab.gray())
    #plt.show()

    experiment = ["poly", "sigmoid"]

    # iterate through all linear
    if "linear" in experiment:
        for c in [0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 1, 10, 20, 100]:
            clf = svm.SVC(kernel='linear', C=c)
            scores = cross_validation.cross_val_score(clf, train_img, np.reshape(train["tr_labels"], (n_images, )) , cv=10)
            print scores.mean()

    # iterate through all poly
    if "poly" in experiment:
        for deg in xrange(1, 3):
            clf = svm.SVC(kernel='poly', degree=deg)
            scores = cross_validation.cross_val_score(clf, train_img, np.reshape(train["tr_labels"], (n_images, )) , cv=250)
            print "POLY: ", deg,  " MAX: ", scores.max(), " MIN: ", scores.min(), " MEAN: ", scores.mean()

    # iterate through all sigmoid
    if "sigmoid" in experiment:
        clf = svm.SVC(kernel='sigmoid')
        scores = cross_validation.cross_val_score(clf, train_img, np.reshape(train["tr_labels"], (n_images, )) , cv=10)
        print scores.mean()
    return clf

def classify(classifier, samples):
    return classifier.predict(samples)

def classify_pub_test(classifier):
    test = scipy.io.loadmat('public_test_images.mat')
    print test
    print test["public_test_images"].shape
    #return classifier.predict(samples)


def main():
    classifier = train_SVM()
    #classify_pub_test(None)

if __name__ == '__main__':
    main()
