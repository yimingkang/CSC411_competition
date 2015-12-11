# file_lsts_svm_no_filter = ['submit_svm_poly2.csv']
file_lsts_svm_no_filter = ['./submit_combine_no_preprocess.csv']
file_lsts_svm = ['./submit_no_normalized_linear.csv', './submit_no_normalized_poly2.csv']
file_lsts_nn = ['./ENV/submit_nn_sing_layer_90_units_100iter_non_normalized.csv', './ENV/submit_nn_sing_layer_90_units_100iter_normalized.csv']
# file_lsts_cnn = ['./ENV/submit_cnn_with_normalization_2.csv', './ENV/submit_cnn_with_normalization_3.csv', './ENV/submit_cnn_with_normalization.csv']
rbf = ['./submit_non-normalized_rbf10.csv']
rbf_normalized = ['./submit_filtered_rbf10.csv']
file_lsts = rbf_normalized + rbf + file_lsts_nn
# file_lsts = file_lsts_cnn
csv_name = file_lsts[0]
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

with open('submit_combine_with_rbf10_2.csv', 'w') as f:
    f.write('Id,Prediction\n')
    index = 1
    for pred in result_lst:
        f.write('%d,%d\n'%(index, pred))
        index += 1
    while index<=1253:
        f.write('%d,0\n'%(index))
        index+=1

print '+'*20
print "Total number of .csv files: %d"%len(file_lsts)
wrong_prediction_files_sorted = sorted(wrong_prediction_files.items(), key=lambda x: x[1])
print wrong_prediction_files_sorted
for wpf in wrong_prediction_files:
    print wpf, wrong_prediction_files[wpf]
print '+'*20
# print predictions.keys()
# print predictions['submit_mac_aaron.csv'][0]
# print predictions['submit_mac_aaron.csv'][417]
# print predictions['submit_mac_aaron.csv'][418]
