import os
import sys
if sys.version_info.major==2:
    import cPickle
else:
    import pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt

from py2py3 import *
import loader

def document_length_analysis(data_folders):
    document_len_list=[]
    summary_len_list=[]

    file_num=0
    for data_folder in data_folders:
        for file_name in os.listdir(data_folder):
            if file_name.split('.')[-1] in ['summary']:
                document_info=loader.parse_document(file_name=data_folder+os.sep+file_name, replace_entity=True)
                if document_info==None:
                    continue
                file_num+=1
                sys.stdout.write('Detected %d files\r'%file_num)                

                document=document_info['sentences']
                summary=document_info['highlights']
                document_len=[len(sent.split(' '))+2 for sent in document]
                summary_len=[len(sent.split(' '))+2 for sent in summary]
                document_len=min(2000, np.sum(document_len)+2)
                summary_len=min(200, np.sum(summary_len)+2)
                document_len_list.append(document_len)
                summary_len_list.append(summary_len)
    print('File loading procedure is completed!') 

    bins=np.arange(0,2000)
    plt.hist(document_len_list, bins=bins, histtype='bar', color='g', edgecolor='g')
    plt.xlabel('document length in tokens')
    plt.ylabel('num of documents')
    plt.show()

    plt.clf()
    bins=np.arange(0,200)
    plt.hist(summary_len_list, bins=bins, histtype='bar', color='g', edgecolor='g')
    plt.xlabel('summary length in tokens')
    plt.ylabel('num of documents')
    plt.show()

    cPickle.dump({'document_len_list':document_len_list, 'summary_len_list':summary_len_list}, open('document_len.pkl','wb'))

def document_important_region_analysis(data_folders):
    total_num=np.zeros(2000, dtype=np.int)
    important_num=np.zeros(2000, dtype=np.int)

    file_num=0
    for data_folder in data_folders:
        for file_name in os.listdir(data_folder):
            if file_name.split('.')[-1] in ['summary']:
                document_info=loader.parse_document(file_name=data_folder+os.sep+file_name, replace_entity=True)
                if document_info==None:
                    continue
                file_num+=1
                sys.stdout.write('Detected %d files\r'%file_num)

                document=document_info['sentences']
                labels=document_info['labels']

                assert len(document)==len(labels), 'file=%s, len(document)=%d, len(labels)=%d'%(
                    data_folder+os.sep+file_name, len(document), len(labels))

                pt_index=1
                for sentence, label in zip(document, labels):
                    tokens=sentence.split(' ')
                    total_num[min(pt_index, 1999)]+=1
                    if label==1:
                        important_num[min(pt_index, 1999)]+=1
                    for idx in xrange(len(tokens)):
                        pt_index+=1
                        total_num[min(pt_index, 1999)]+=1
                        if label==1:
                            important_num[min(pt_index, 1999)]+=1
                    total_num[min(pt_index, 1999)]+=1
                    if label==1:
                        important_num[min(pt_index, 1999)]+=1
    print('File loading procedure is completed!')

    plt.plot(np.arange(1,2000), total_num[1:], color='g', label='all tokens')
    plt.plot(np.arange(1,2000), important_num[1:], color='b', label='important tokens')
    plt.xlabel('position of tokens')
    plt.ylabel('number of documents')
    plt.legend()
    plt.show()

    cPickle.dump({'total_num':total_num, 'important_num':important_num}, open('important_region.pkl','wb'))

if len(sys.argv)<2:
    print('Usage: python data_analysis.py [<data_folder>...]')
    exit(0)

document_length_analysis(sys.argv[1:])
# document_important_region_analysis(sys.argv[1:])
