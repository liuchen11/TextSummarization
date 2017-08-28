import os
import sys
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
sys.path.insert(0,'util')
from py2py3 import *
import numpy as np
import random
import matplotlib.pyplot as plt

'''
>>> dict_file: str, dictionary file
>>> embedding_file: str, embedding file
'''
def plot_word_frequency_distribution(dict_file, embedding_file):
    available_word_list=get_available_word_list(embedding_file,format='text') if embedding_file!=None else None
    notes=[]                # list of (frequency, all_words, available_words)
    with open(dict_file,'r') as fopen:
        lines=fopen.readlines()
        lines=map(lambda x:x[:-1] if x[-1]=='\n' else x, lines)
        for idx,line in enumerate(lines[1:]):
            sys.stdout.write('%d/%d words loaded - %.1f%%\r'%(idx,len(lines)-1,float(idx)/float(len(lines)-1)*100))
            parts=line.split(' ')
            word,frequency=parts[1],int(parts[2])
            if len(notes)==0:
                notes.append([frequency,1,0])
            elif notes[-1][0]!=frequency:
                notes.append([frequency,notes[-1][1]+1,notes[-1][2]])
            else:
                notes[-1][1]+=1
            if available_word_list!=None and word in available_word_list:
                notes[-1][2]+=1
    word_total_num=notes[-1][1]
    notes=sorted(notes,lambda x,y:-1 if x[0]<y[0] else 1)
    num_of_word_above_freq=map(lambda x:x[1],notes)
    available_rate=map(lambda x:float(x[2])/float(x[1])*100,notes)
    frequency_list=map(lambda x:x[0],notes)

    fig,ax1=plt.subplots()
    ax1.set_xscale('log',nonposx='clip')
    ax1.plot(frequency_list,num_of_word_above_freq,color='r')
    ax1.set_xlabel('word frequency')
    ax1.set_ylabel('num of words')

    ax2=ax1.twinx()
    ax2.plot(frequency_list,available_rate,color='g')
    ax2.set_ylabel('percentage of available embeddings')

    fig.tight_layout()
    plt.show()

'''
>>> file: str, embedding information
>>> format: str, in which format is word embeddings stored
'''
def get_available_word_list(file,format):
    if not format in ['text','bin']:
        raise ValueError('Invalid format %s'%format)

    word_list=[]
    if format=='text':
        lines=open(file,'r').readlines()
        lines=map(lambda x: x[:-1] if x[-1]!='\n' else x,lines)
        for idx,line in enumerate(lines):
            sys.stdout.write('Loading embeddings %d/%d - %.1f%%\r'%(idx+1,len(lines),float(idx+1)/float(len(lines))*100))
            parts=line.split(' ')
            word=parts[0]
            word_list.append(word)
    elif format=='bin':
        with open(file,'r') as fopen:
            header=fopen.readline()
            vocabulary_size,dimension=map(int,header.split())
            binary_length=np.dtype('float32').itemsize*dimension
            for word_idx in xrange(vocabulary_size):
                sys.stdout.write('Loading embeddings %d/%d - %.1f%%\r'%(word_idx+1,vocabulary_size,float(word_idx+1)/float(vocabulary_size)*100))
                word=[]
                while True:
                    ch=fopen.read(1)
                    if ch==' ':
                        word=''.join(word)
                        break
                    else:
                        word.append(ch)
                word_list.append(word)
    print('vocabulary loaded from %s'%file)
    return word_list

'''
>>> file_or_folder_list: str, files or folders that contain documents
'''
def plot_document_length_distribution(file_or_folder_list):
    file_list=[]
    documents_length_distribution=[]
    for file_or_folder in file_or_folder_list:
        if os.path.isdir(file_or_folder):
            for file in os.listdir(file_or_folder):
                if file.split('.')[-1] in ['info',]:
                    file_list.append(file_or_folder+os.sep+file)
        elif os.path.isfile(file_or_folder):
            if file_or_folder.split('.')[-1] in ['info',]:
                file_list.append(file_or_folder)
    print('detected %d files in total'%len(file_list))

    for idx,file in enumerate(file_list):
        sys.stdout.write('%d/%d file loaded - %.1f%%\r'%(idx,len(file_list),float(idx)/float(len(file_list))*100))
        with open(file,'r') as fopen:
            document_length=int(fopen.readline())
            documents_length_distribution.append(document_length)

    plt.yscale('log',nonposx='clip')
    bins=np.arange(0,np.max(documents_length_distribution)+10,10)
    plt.hist(documents_length_distribution,bins=bins,histtype='bar',color='g',edgecolor='g')
    plt.xlabel('num of sentences')
    plt.ylabel('num of documents')
    plt.show()

def plot_sentence_length_distribution(file_or_folder_list):
    file_list=[]
    sentence_length_distribution=[]
    for file_or_folder in file_or_folder_list:
        if os.path.isdir(file_or_folder):
            for file in os.listdir(file_or_folder):
                if file.split('.')[-1] in ['info',]:
                    file_list.append(file_or_folder+os.sep+file)
        elif os.path.isfile(file_or_folder):
            if file_or_folder.split('.')[-1] in ['info',]:
                file_list.append(file_or_folder)
    print('detected %d files in total'%len(file_list))

    for idx,file in enumerate(file_list):
        sys.stdout.write('%d/%d file loaded - %.1f%%\r'%(idx,len(file_list),float(idx)/float(len(file_list))*100))
        with open(file,'r') as fopen:
            document_length=int(fopen.readline())
            for line_idx in xrange(document_length):
                parts=fopen.readline().split(',')
                sentence_length_distribution.append(len(parts))
    print('detected %d sentences in total'%len(sentence_length_distribution))

    plt.yscale('log',nonposx='clip')
    bins=np.arange(0,np.max(sentence_length_distribution)+10,10)
    plt.hist(sentence_length_distribution,bins=bins,histtype='bar',color='g',edgecolor='g')
    plt.xlabel('num of tokens')
    plt.ylabel('num of sentences')
    plt.show()

def plot_sentence_label_distribution(file_or_folder_list,document_length_threshold,class_num):
    file_list=[]
    sentence_length_distribution=[]
    for file_or_folder in file_or_folder_list:
        if os.path.isdir(file_or_folder):
            for file in os.listdir(file_or_folder):
                if file.split('.')[-1] in ['info',]:
                    file_list.append(file_or_folder+os.sep+file)
        elif os.path.isfile(file_or_folder):
            if file_or_folder.split('.')[-1] in ['info',]:
                file_list.append(file_or_folder)
    print('detected %d files in total'%len(file_list))

    sentence_num_by_class_original=[0 for idx in xrange(class_num)]
    sentence_num_by_class_filtered=[0 for idx in xrange(class_num)]
    for idx,file in enumerate(file_list):
        sys.stdout.write('%d/%d file loaded - %.1f%%\r'%(idx+1,len(file_list),float(idx+1)/float(len(file_list))*100))
        with open(file,'r') as fopen:
            lines=fopen.readlines()
            lines=map(lambda x: x if x[-1]!='\n' else x[:-1],lines)
            labels=map(int,lines[-1].split(','))
            if len(labels)<=document_length_threshold:
                for label in labels:
                    sentence_num_by_class_original[label]+=1
                    sentence_num_by_class_filtered[label]+=1
            else:
                max_pad=len(labels)-document_length_threshold
                begin_idx=random.randint(0,max_pad)
                end_idx=begin_idx+document_length_threshold
                for label_idx,label in enumerate(labels):
                    sentence_num_by_class_original[label]+=1
                    if label_idx>=begin_idx and label_idx<end_idx:
                        sentence_num_by_class_filtered[label]+=1
    print('')
    print('Original Text:')
    for class_idx in xrange(class_num):
        print('Category %d: %d sentences'%(class_idx,sentence_num_by_class_original[class_idx]))
    print('Total %d sentences'%(np.sum(sentence_num_by_class_original)))
    print('Filtered Text of maximum %d sentences'%(document_length_threshold))
    for class_idx in xrange(class_num):
        print('Category %d: %d sentences'%(class_idx,sentence_num_by_class_filtered[class_idx]))
    print('Total %d sentences'%(np.sum(sentence_num_by_class_filtered)))

def plot_sentence_label_length_distribution(file_or_folder_list,max_length,class_num):
    file_list=[]
    sentence_length_distribution=[]
    for _ in xrange(class_num):
        sentence_length_distribution.append([0 for idx in xrange(max_length)])

    for file_or_folder in file_or_folder_list:
        if os.path.isdir(file_or_folder):
            for file in os.listdir(file_or_folder):
                if file.split('.')[-1] in ['info',]:
                    file_list.append(file_or_folder+os.sep+file)
        elif os.path.isfile(file_or_folder):
            if file_or_folder.split('.')[-1] in ['info',]:
                file_list.append(file_or_folder)
    print('detected %d files in total'%len(file_list))

    for idx,file in enumerate(file_list):
        sys.stdout.write('%d/%d file loaded - %.1f%%\r'%(idx+1,len(file_list),float(idx+1)/float(len(file_list))*100))
        lines=open(file,'r').readlines()
        lines=map(lambda x: x if x[-1]!='\n' else x[:-1],lines)
        num_of_sentence=int(lines[0])
        sentence_length_list=map(lambda x:len(x.split(',')),lines[1:-1])
        label_list=map(int,lines[-1].split(','))
        for sentence_length,label in zip(sentence_length_list,label_list):
            sentence_length=min(max_length,sentence_length)
            sentence_length_distribution[label][sentence_length-1]+=1

    sentence_length_distribution=np.array(sentence_length_distribution)
    total_num_sentences=[0 for _ in xrange(max_length)]
    for length in xrange(max_length):
        total_num_sentences[length]=np.sum(sentence_length_distribution[:,length])
    for length in xrange(max_length):
        if total_num_sentences[length]==0:
            continue
        out_str='length = %d\t'%(length+1)
        for class_idx in xrange(class_num):
            num=sentence_length_distribution[class_idx,length]
            prop=float(num)/float(total_num_sentences[length])*100
            out_str+='%d: %d(%.1f%%)\t'%(class_idx,num,prop)
        print(out_str)

def plot_sentence_label_position_distribution(file_or_folder_list,max_num,class_num):
    file_list=[]
    sentence_position_distribution=[]
    for _ in xrange(class_num):
        sentence_position_distribution.append([0 for idx in xrange(max_num)])

    for file_or_folder in file_or_folder_list:
        if os.path.isdir(file_or_folder):
            for file in os.listdir(file_or_folder):
                if file.split('.')[-1] in ['info',]:
                    file_list.append(file_or_folder+os.sep+file)
        elif os.path.isfile(file_or_folder):
            if file_or_folder.split('.')[-1] in ['info',]:
                file_list.append(file_or_folder)
    print('detected %d files in total'%len(file_list))

    for idx,file in enumerate(file_list):
        sys.stdout.write('%d/%d file loaded - %.1f%%\r'%(idx+1,len(file_list),float(idx+1)/float(len(file_list))*100))
        lines=open(file,'r').readlines()
        label_list=map(int,lines[-1].split(','))
        for pos,label in enumerate(label_list):
            pos=min(max_num-1,pos)
            try:
                sentence_position_distribution[label][pos]+=1
            except:
                print('file=%s,label=%d,pos=%d'%(file,label,pos))
                exit(0)

    sentence_position_distribution=np.array(sentence_position_distribution)
    total_num_sentences=[0 for _ in xrange(max_num)]
    for pos in xrange(max_num):
        total_num_sentences[pos]=np.sum(sentence_position_distribution[:,pos])
    for pos in xrange(max_num):
        if total_num_sentences[pos]==0:
            continue
        out_str='pos = %d\t'%(pos+1)
        for class_idx in xrange(class_num):
            num=sentence_position_distribution[class_idx,pos]
            prop=float(num)/float(total_num_sentences[pos])*100
            out_str+='%d: %d(%.1f%%)\t'%(class_idx,num,prop)
        print(out_str)

if __name__=='__main__':
#     if len(sys.argv)!=3:
#         print('python data_analysis.py <dict_file> <embedding_file>')
#         exit(0)
#     plot_word_frequency_distribution(dict_file=sys.argv[1],embedding_file=sys.argv[2])

    # if len(sys.argv)<2:
    #     print('python data_analysis.py <document_folder_list> ...')
    #     exit(0)
    # plot_document_length_distribution(sys.argv[1:])
    # plot_sentence_length_distribution(sys.argv[1:])

    if len(sys.argv)<2:
        print('Usage: python data_analysis.py <document_folder_list>')
        exit(0)

    # plot_sentence_label_length_distribution(sys.argv[1:],100,3)
    plot_sentence_label_position_distribution(sys.argv[1:],60,3)
