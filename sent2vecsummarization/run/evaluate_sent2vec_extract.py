import os
import sys
sys.path.insert(0,'util')
if sys.version_info.major==2:
    import cPickle
else:
    import pickle as cPickle
import rouge
import numpy as np

from py2py3 import *

'''
>>> draw the position distribution of the sentence selected
'''
def position_distribution(pkl_file, output_file, max_len=60):
    position_dist_list=[0, ]*max_len
    result_list=cPickle.load(open(pkl_file, 'rb'))
    for result_per_document in result_list:
        for sentence_selected in result_per_document['result']:
            sentence_idx=sentence_selected[2]
            sentence_idx=min(max_len-1, sentence_idx)
            position_dist_list[sentence_idx]+=1

    print('Position Distribution:')
    for idx, number in enumerate(position_dist_list):
        print('%02d >>> %04d'%(idx+1, number))
    cPickle.dump(position_dist_list, open(output_file, 'wb'))

    return position_dist_list

'''
>>> draw the ground truth label distribution of the sentence selected
'''
def label_distribution(pkl_file, output_file,  class_num=3):
    label_dist_list=[0, ]*class_num
    result_list=cPickle.load(open(pkl_file, 'rb'))
    for result_per_document in result_list:
        ground_truth_list=result_per_document['ground_truth']
        for sentence_selected in result_per_document['result']:
            sentence_idx=sentence_selected[2]
            label_idx=ground_truth_list[sentence_idx]
            label_dist_list[label_idx]+=1
    print('Label Distribution:')
    for idx, number in enumerate(label_dist_list):
        print('%02d >>> %04d (%.1f%%)'%(idx, number, float(number)/float(np.sum(label_dist_list))*100))
    cPickle.dump(label_dist_list, open(output_file, 'wb'))

    return label_dist_list

'''
>>> calculate the rouge score
'''
def calc_rouge(pkl_file, output_file):
    r=rouge.Rouge()

    rouge_result={'summary':{'results':{'rouge-1':{'p':0.0, 'r':0.0, 'f':0.0}, 'rouge-2':{'p':0.0, 'r':0.0, 'f':0.0},
        'rouge-l':{'p':0.0, 'r':0.0, 'f':0.0}},'valid_documents':0}, 'details':[]}
    result_list=cPickle.load(open(pkl_file, 'rb'))
    for idx, result_per_document in enumerate(result_list):
        sys.stdout.write('Loading documents %d/%d=%.1f%%\r'%(idx+1, len(result_list),
            float(idx+1)/float(len(result_list))*100))
        summary_reference=result_per_document['summary']
        selected_sentence_list=[]

        for sentence_selected in result_per_document['result']:
            selected_sentence_list.append(sentence_selected[0])

        summary_extracted=' '.join(selected_sentence_list)
        summary_reference=summary_reference.decode('latin1',errors='ignore').encode('ascii',errors='ignore')
        summary_extracted=summary_extracted.decode('latin1',errors='ignore').encode('ascii',errors='ignore')
        single_result=r.get_scores(summary_reference, summary_extracted)[0]
        rouge_result['details'].append({'results':single_result, 'reference':summary_reference,
            'decode':summary_extracted})
        for metric in ['p', 'r', 'f']:
            rouge_result['summary']['results']['rouge-1'][metric]+=single_result['rouge-1'][metric]
            rouge_result['summary']['results']['rouge-2'][metric]+=single_result['rouge-2'][metric]
            rouge_result['summary']['results']['rouge-l'][metric]+=single_result['rouge-l'][metric]
        rouge_result['summary']['valid_documents']+=1

    print('All documents are loaded completely!!')
    if rouge_result['summary']['valid_documents']!=0:
        rouge_result['summary']['results']['rouge-1'][metric]/=rouge_result['summary']['valid_documents']
        rouge_result['summary']['results']['rouge-2'][metric]/=rouge_result['summary']['valid_documents']
        rouge_result['summary']['results']['rouge-l'][metric]/=rouge_result['summary']['valid_documents']

    cPickle.dump(rouge_result, open(output_file, 'wb'))

if __name__=='__main__':

    if len(sys.argv)<3:
        print('Usage: python evaluate_sent2vec_extract.py <pkl_file> <saved_file>')
        exit(0)

    label_distribution(pkl_file=sys.argv[1], output_file=sys.argv[2])
