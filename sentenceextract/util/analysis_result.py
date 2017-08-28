import os
import sys
if sys.version_info.major==2:
    import cPickle
else:
    import pickle as cPickle
    xrange=range

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

if len(sys.argv)!=3:
    print('Usage: python analysis_result.py <pkl file> <n_top>')
    exit(0)

pkl_file=sys.argv[1]
n_top=int(sys.argv[2])

# The format of the pkl content:
# list of {'file_name':#FILE_NAME#, 'result':[(idx, important_idx)], 'ground_truth':[sequence of labels]}
info=cPickle.load(open(pkl_file, 'rb'))
correct_num=0
total_num=0

for idx, document_info in enumerate(info):
    result_list=document_info['result']
    ground_truth=document_info['ground_truth']

    if sys.version_info.major==2:
        result_list=sorted(result_list, lambda x,y: 1 if x[1]<y[1] else -1)
    else:
        result_list=sorted(result_list, key=lambda x:x[1], reverse=True)
    for sentence_idx, importance in result_list[:n_top]:
        true_label=ground_truth[sentence_idx]
        total_num+=1
        if true_label==1:
            correct_num+=1

    sys.stdout.write('Loading %d/%d=%.1f%%, percentage = %d/%d=%.1f%%\r'%(idx+1, len(info),
        float(idx+1)/float(len(info))*100, correct_num, total_num, float(correct_num)/float(total_num)*100))
    sys.stdout.flush()

print('')

