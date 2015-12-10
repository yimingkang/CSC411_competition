csv_name = 'submit_svm_linear.csv'
# file_lsts = ['submit_svm_poly2.csv', 'submit_svm_linear.csv',  'submit_svm_poly3.csv', 'submit_non-normalized_poly2.csv', 'submit_normalized_linear.csv', 'submit_normalized_poly2.csv']
file_lsts = ['submit_svm_poly2.csv', 'submit_svm_linear.csv',  'submit_svm_poly3.csv', './Submissions/prediction_ensemble.csv', './Submissions/preddiction_nn_hu_15_gabor_pca_100.csv', './Submissions/prediction_final_multiclass_10000_0006.csv']
predictions=dict()

wrong_prediction_files=dict()
for file_to_read in file_lsts:
    current_predictions = []
    with open(file_to_read, 'r') as f:
        for line in f:
            # print line.strip().split(',')[1]
            current_predictions.append(line.strip().split(',')[1])
    predictions[file_to_read] = current_predictions[1:]

result_lst = []
for i in xrange(len(predictions[csv_name])):
    freq = dict()
    for file_to_read in file_lsts:
        prediction  = predictions[file_to_read][i]
        if prediction in freq:
            freq[prediction] += 1
        else:
            freq[prediction] = 1
    p = sorted(freq.items(), key=lambda x: x[1])[-1][0]
    if freq[p] != len(predictions):
        wrong_preds = [failed_pred for failed_pred in file_lsts if predictions[failed_pred][i]!=p]
        for w_p in wrong_preds:
            if w_p in wrong_prediction_files:
                wrong_prediction_files[w_p]+=1
            else:
                wrong_prediction_files[w_p] = 1
        print "i: %d;Expect: %s. But:"%(i, p),
        print [(fi, predictions[fi][i]) for fi in wrong_preds]
    result_lst.append(int(p))

with open('submit_combine_no_preprocess.csv', 'w') as f:
    f.write('Id,Prediction\n')
    index = 1
    for pred in result_lst:
        f.write('%d,%d\n'%(index, pred))
        index += 1
    while index<=1253:
        f.write('%d,0\n'%(index))
        index+=1

print '+'*20
for wpf in wrong_prediction_files:
    print wpf, wrong_prediction_files[wpf]
print '+'*20
# print predictions.keys()
# print predictions['submit_mac_aaron.csv'][0]
# print predictions['submit_mac_aaron.csv'][417]
# print predictions['submit_mac_aaron.csv'][418]
