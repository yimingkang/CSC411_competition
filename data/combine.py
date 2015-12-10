file_lsts = ['submit_mac_aaron.csv', 'submit_svm_linear.csv',  'submit_non-normalized_linear.csv', 'submit_non-normalized_poly2.csv', 'submit_normalized_linear.csv', 'submit_normalized_poly2.csv']
predictions=dict()
for file_to_read in file_lsts:
    current_predictions = []
    with open(file_to_read, 'r') as f:
        for line in f:
            # print line.strip().split(',')[1]
            current_predictions.append(line.strip().split(',')[1])
    predictions[file_to_read] = current_predictions[1:]

result_lst = []
for i in xrange(len(predictions['submit_mac_aaron.csv'])):
    freq = dict()
    for file_to_read in file_lsts:
        prediction  = predictions[file_to_read][i]
        if prediction in freq:
            freq[prediction] += 1
        else:
            freq[prediction] = 1
    p = sorted(freq.items(), key=lambda x: x[1])[-1][0]
    result_lst.append(int(p))

with open('submit_combine_new.csv', 'w') as f:
    f.write('Id,Prediction\n')
    index = 1
    for pred in result_lst:
        f.write('%d,%d\n'%(index, pred))
        index += 1
    while index<=1253:
        f.write('%d,0\n'%(index))
        index+=1
# print predictions.keys()
print predictions['submit_mac_aaron.csv'][0]
print predictions['submit_mac_aaron.csv'][417]
print predictions['submit_mac_aaron.csv'][418]
