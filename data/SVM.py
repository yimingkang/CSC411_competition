def preproc():
    # first compute the average brightness
    test = scipy.io.loadmat('labeled_images.mat')['tr_images']
    print "Shape of test images is : ", test.shape
    (x, y, n_images) = test.shape
    train_img = np.reshape(np.swapaxes(test, 0, 2), (n_images, x * y))
    print "Shape of transformed image is : ", train_img.shape
    avg_array = []
    for i in xrange(n_images):
        avg_array.append(train_img[i].mean())

    # the histogram of the data
    n, bins, patches = plt.hist(avg_array, 50, normed=1, facecolor='green', alpha=0.75)

    plt.xlabel('Brightness')
    plt.ylabel('Probability')
    plt.title("Avg brightness distribution")
    plt.grid(True)

    plt.show()


def train_SVM():
    train = scipy.io.loadmat('labeled_images.mat')

    print "Shape of tr_images is: ", train["tr_images"].shape
    (x, y, n_images) = train["tr_images"].shape

    train_img = np.reshape(np.swapaxes(train["tr_images"], 0, 2), (n_images, x * y))
    #plt.imshow(np.swapaxes(np.reshape(train_img[0], (y, x)), 0, 1), cmap=pylab.gray())
    #plt.show()

    clf = svm.SVC(kernel='poly', degree=2)
    clf.fit(train_img, np.reshape(train["tr_labels"], (n_images, )))
    scores = cross_validation.cross_val_score(clf, train_img, np.reshape(train["tr_labels"], (n_images, )) , cv=3)

    print scores
    print scores.mean()
    return clf

def classify(classifier, samples):
    return classifier.predict(samples)

def classify_pub_test(classifier):
    test = scipy.io.loadmat('public_test_images.mat')
    print test
    print test["public_test_images"].shape
    (x, y, n_images) = test["public_test_images"].shape
    test_img = np.reshape(np.swapaxes(test["public_test_images"], 0, 2), (n_images, x * y))
    return classifier.predict(test_img)

def main():
    # preproc()
    print "Training SVM..."
    classifier = train_SVM()
    print "Classifing..."
    classify_result = classify_pub_test(classifier)
    cls_res_list = list(classify_result)
    print cls_res_list
    with open('submit.csv', 'w') as f:
        f.write('Id,Prediction\n')
        index = 1
        for pred in cls_res_list:
            f.write('%d,%d\n'%(index, pred))
            index += 1
        while index<=1253:
            f.write('%d,0\n'%(index))
            index+=1

if __name__ == '__main__':
    print "Importing libs..."
    import scipy.io
    import pylab
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import svm
    from sklearn import cross_validation
    print "Done!"
    main()
