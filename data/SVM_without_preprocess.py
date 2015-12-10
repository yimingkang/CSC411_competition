def train_SVM():
    train = scipy.io.loadmat('labeled_images.mat')

    print "Shape of tr_images is: ", train["tr_images"].shape
    (x, y, n_images) = train["tr_images"].shape
    # (n_images, dim) = train["tr_images"].shape

    train_img = np.reshape(np.swapaxes(train["tr_images"], 0, 2), (n_images, x * y))
    #plt.imshow(np.swapaxes(np.reshape(train_img[0], (y, x)), 0, 1), cmap=pylab.gray())
    #plt.show()

    clf = svm.SVC(kernel='poly', degree=3)
    # clf = svm.SVC(kernel='linear', C=10)
    clf.fit(train_img, np.reshape(train["tr_labels"], (n_images, )))
    # scores = cross_validation.cross_val_score(clf, train_img, np.reshape(train["tr_labels"], (n_images, )) , cv=3)

    # print scores
    # print scores.mean()
    return clf

def classify_pub_test(classifier):
    pub_test = scipy.io.loadmat('public_test_images.mat')
    hid_test = scipy.io.loadmat('hidden_test_images.mat')
    (x, y, n_images) = pub_test["public_test_images"].shape
    test_img = np.reshape(np.swapaxes(pub_test["public_test_images"], 0, 2), (n_images, x * y))
    pub_res = list(classifier.predict(test_img))
    (x, y, n_images) = hid_test["hidden_test_images"].shape
    test_img = np.reshape(np.swapaxes(hid_test["hidden_test_images"], 0, 2), (n_images, x * y))
    hid_res = list(classifier.predict(test_img))
    return pub_res+hid_res

def main():
    # preproc()
    print "Training SVM..."
    classifier = train_SVM()
    print "Classifing..."
    cls_res_list = classify_pub_test(classifier)
    print cls_res_list
    with open('submit_svm_poly3.csv', 'w') as f:
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
