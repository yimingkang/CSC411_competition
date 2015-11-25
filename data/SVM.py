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
    plt.imshow(np.swapaxes(np.reshape(train_img[0], (y, x)), 0, 1), cmap=pylab.gray())
    plt.show()

    clf = svm.SVC()
    #clf = svm.SVC(kernel='linear', C=1)
    scores = cross_validation.cross_val_score(clf, train_img, np.reshape(train["tr_labels"], (n_images, )) , cv=5)
    print scores
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
