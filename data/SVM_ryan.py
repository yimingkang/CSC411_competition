def double_rainbow():
    train_img, train_labels = get_training_data()
    return normalize(equalize_hist(train_img)), train_labels

def normalize(train_img):
    return preprocessing.scale(train_img * 1.0, axis=1)

def equalize_hist(train_img):
    n_images, _ = train_img.shape
    for i in xrange(n_images):
        img = train_img[0]
        train_img[0] = exposure.equalize_hist(img.reshape(32, 32)).ravel()
    return train_img

def get_equalized_data():
    # Load an example image
    train_img, train_labels = get_training_data()
    return equalize_hist(train_img), train_labels 

def get_normalized_data():
    # Load an example image
    train_img, train_labels = get_training_data()
    return normalize(train_img), train_labels

def get_training_data():
    train = scipy.io.loadmat('labeled_images.mat')
    (x, y, n_images) = train["tr_images"].shape
    train_img = np.reshape(np.swapaxes(train["tr_images"], 0, 2), (n_images, x * y))
    return train_img, train['tr_labels']

def train_SVM():
    train_img, train_labels = get_equalized_data()
    n_images, _ = train_img.shape
    experiment = ["linear", "poly", "sigmoid"]

    # iterate through all linear
    if "linear" in experiment:
        for c in [1, 10, 100]:
            clf = svm.SVC(kernel='linear', C=c)
            scores = cross_validation.cross_val_score(clf, train_img, np.reshape(train_labels, (n_images, )) , cv=5)
            print "linear mean: ", scores.mean()

    # iterate through all poly
    if "poly" in experiment:
        for deg in xrange(1, 6):
            clf = svm.SVC(kernel='poly', degree=deg)
            scores = cross_validation.cross_val_score(clf, train_img, np.reshape(train_labels, (n_images, )) , cv=5)
            print "POLY: ", deg,  " MAX: ", scores.max(), " MIN: ", scores.min(), " MEAN: ", scores.mean()

    # iterate through all sigmoid
    if "sigmoid" in experiment:
        clf = svm.SVC(kernel='sigmoid')
        scores = cross_validation.cross_val_score(clf, train_img, np.reshape(train_labels, (n_images, )) , cv=5)
        print "Sigmoid mean: ", scores.mean()
    return clf

def classify(classifier, samples):
    return classifier.predict(samples)

def classify_pub_test(classifier):
    test = scipy.io.loadmat('public_test_images.mat')
    print test
    print test["public_test_images"].shape
    #return classifier.predict(samples)

def show_img(img):
    plt.imshow(img.reshape(32, 32))
    plt.show()

def main():
    #preproc()
    #get_training_data()
    #get_equalized_data()
    classifier = train_SVM()
    #classify_pub_test(None)

if __name__ == '__main__':
    print "Importing libs..."
    import scipy.io
    import pylab
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import svm
    from sklearn import cross_validation
    from sklearn import preprocessing
    from skimage import exposure
    from skimage import data
    print "done!"
    main()
