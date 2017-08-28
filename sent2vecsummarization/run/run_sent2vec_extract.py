import os
import sys
sys.path.insert(0,'./util')
sys.path.insert(0,'./model')
import numpy as np
if sys.version_info.major==2:
    import cPickle
else:
    import pickle as cPickle

from py2py3 import *
import loader
import sent2vec_extract
import xml_parser

if __name__=='__main__':

    if len(sys.argv)!=2:
        print('Usage: python run_sent2vec_extract.py <config>')
        exit(0)

    hyper_params=xml_parser.parse(sys.argv[1],flat=False)
    sent2vec_extract_params=hyper_params['sent2vec_extract']
    sort_by_order=True if 'sort_by_order' not in hyper_params else hyper_params['sort_by_order']
    saved_marks=True if 'saved_marks' not in hyper_params else hyper_params['saved_marks']

    folder2scan=hyper_params['folder2scan']
    output_file=hyper_params['output_file']
    label_policy='min' if not 'label_policy' in hyper_params else hyper_params['label_policy']

    file_list=[]
    for file_name in os.listdir(folder2scan):
        if os.path.isfile(folder2scan+os.sep+file_name) and file_name.split('.')[-1] in ['summary']:
            file_list.append(folder2scan+os.sep+file_name)

    print('There are %d files detected in total!'%len(file_list))
    solver=sent2vec_extract.Sent2VecExtract(sent2vec_extract_params)

    document_list=[]
    summary_list=[]
    valid_file_list=[]
    label_list=[]
    for idx,file_name in enumerate(file_list):
        sys.stdout.write('%d/%d=%.1f%% document loaded\r'%(idx+1, len(file_list),
            float(idx+1)/float(len(file_list))*100))
        sys.stdout.flush()
        parsed_information=loader.parse_document(file_name)
        if parsed_information!=None:
            document_list.append('\n'.join(parsed_information['sentences']))
            summary_list.append('\n'.join(parsed_information['highlights']))
            valid_file_list.append(file_name)
            label_list.append(parsed_information['labels'])

    file_list=valid_file_list
    key_sentences=solver.extract(document_list, return_idx=True, sort_by_order=sort_by_order, saved_marks=saved_marks)
    assert len(file_list)==len(document_list), 'len(file_list)=%d len(document_list)=%d'%(len(file_list),len(document_list))
    assert len(file_list)==len(summary_list), 'len(file_list)=%d len(summary_list)=%d'%(len(file_list),len(document_list))
    assert len(file_list)==len(key_sentences), 'len(file_list)=%d len(key_sentences)=%d'%(len(file_list),len(key_sentences))
    assert len(file_list)==len(label_list), 'len(file_list)=%d len(label_list)=%d'%(len(file_list),len(label_list))
    to_save=[]
    for file_name, document, summary, result, labels in zip(file_list, document_list, summary_list, key_sentences, label_list):
        to_save.append({'file_name':file_name, 'document':document, 'summary':summary, 'result':result, 'ground_truth':labels})
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    cPickle.dump(to_save, open(output_file, 'wb'))
    print('Completed, results are saved in %s'%output_file)


