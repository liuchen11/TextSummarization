import os
import sys
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
sys.path.insert(0,'util')
from py2py3 import *
import random
import numpy as np

'''
>>> text manager
'''
class text_manager(object):

    '''
    >>> Constructor
    '''
    def __init__(self,hyper_params):
        self.fix_dict=hyper_params['fix_dict'] if 'fix_dict' in hyper_params else False
        self.word_list=hyper_params['word_list'] if self.fix_dict==True else []

    '''
    >>> generate the input of the network
    >>> file_name: str, name of text file
    >>> network: model.network, neural network model
    >>> output: indices and masks
        >>> if self.fix_dict=True: words not occurred in the dict is marked len(self.word_list), 
        paddings are marked len(self.word_list)+1
        >>> if self.fix_dict=False: paddings are marked len(self.word_list)
    '''
    def gen_network_input(self,file_name,network):
        document_length=network.sequence_num
        sentence_length=network.sequence_length

        input_matrix=np.zeros([1,document_length,sentence_length],dtype=np.int)
        input_matrix.fill(len(self.word_list)+1 if self.fix_dict else -1)
        masks=np.zeros([1,document_length],dtype=np.int)

        sentences=open(file_name,'r').readlines()
        sentences=map(lambda x:x.split(' ') if x[-1]!='\n' else x[:-1].split(' '), sentences)

        if len(sentences)>document_length:
            print('Warning: this document contains too much sentences, only first %s sentences are considered')
            sentences=sentences[:document_length]

        sentence_num=len(sentences)
        masks[0,:sentence_num].fill(1)
        for idx,sentence in enumerate(sentences):
            word_idx=map(self.word_lookup,sentence)
            word_idx=word_idx[:sentence_length]
            input_matrix[0,idx,:len(word_idx)]=word_idx

        if self.fix_dict==False:
            fill_padding=np.vectorize(lambda x:x if x!=-1 else len(self.word_list))
            input_matrix=fill_padding(input_matrix)

        return input_matrix,masks

    '''
    >>> word lookup
    '''
    def word_lookup(self,word):
        if self.fix_dict:
            if self.word_list.count(word)==0:
                return len(self.word_list)
            else:
                return self.word_list.index(word)
        else:
            if self.word_list.count(word)==0:
                self.word_list.append(word)
            return self.word_list.index(word)

