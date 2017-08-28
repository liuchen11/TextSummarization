import os
import sys

# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
    from builtins import input
    import pickle as cPickle
else:
    input=raw_input
    import cPickle
sys.path.insert(0,'./util')
from py2py3 import *
import tensorflow as tf
import numpy as np
import shutil

sys.path.insert(0,'./model')
sys.path.insert(0,'./util')

import sentence_extractor
import data_generator
import xml_parser

class laucher(object):

    '''
    >>> Construction function
    '''
    def __init__(self,hyper_params):
        # Basic configuration
        self.model_tag=hyper_params['model_tag']


        # Specify network type and construct the model
        if not self.model_tag.lower() in ['sentence_extractor','se','s_e']:
            raise ValueError('Unrecognized model tag: %s'%self.model_tag)

        network_params=hyper_params[self.model_tag]
        if self.model_tag.lower() in ['sentence_extractor','se','s_e']:
            self.model_tag='sentence extractor'
            self.model=sentence_extractor.sentence_extractor(network_params)
        else:
            raise Exception('Failed to construct the network, model_tag: %s'%self.model_tag)

        # Load the weight for network
        self.model2load=hyper_params['model2load']
        #self.model.train_validate_test_init()
        #self.model.load_params(model2load)

        # Load the data_generator
        data_generator_params=hyper_params['data_generator_params']
        word_file=hyper_params['word_file']
        entity_file=hyper_params['entity_file']
        format=hyper_params['format'] if 'format' in hyper_params else None
        self.generator=data_generator.data_generator(data_generator_params)
        self.generator.load(word_file,entity_file,format)

        # Other configurations
        self.folder2store=hyper_params['folder2store']
        #if os.path.exists(self.folder2store):
        #    raise ValueError('Can not use a existing folder to store temporary files: %s'%self.folder2store)
        if not os.path.exists(self.folder2store):
            os.makedirs(self.folder2store)
        self.n_top=hyper_params['n_top'] if 'n_top' in hyper_params else 5
        print('A solver based on model %s already constructed'%self.model_tag)

    def start(self,loaded_params=[]):
        if self.model.sess==None:
            self.model.train_validate_test_init()
            self.model.load_params(self.model2load,loaded_params)

    def end(self):
        self.model.train_validate_test_end()

    '''
    >>> generate the summary
    '''
    def run(self, in_file):
        top_sentence_list=self.model.do_summarization(file_list=[in_file,],folder2store=self.folder2store,
            data_generator=self.generator,n_top=self.n_top)[0]
        # sort by the sentence order in original document
        if sys.version_info.major==2:
            top_sentence_list=sorted(top_sentence_list,lambda x,y: -1 if x[0]<y[0] else 1)
        else:
            top_sentence_list=sorted(top_sentence_list,key=lambda x:x[0],reverse=False)
        out_str=''
        for sentence_idx,prediction,sentence in top_sentence_list:
            out_str+='[%d] (%.3f) %s\n'%(sentence_idx,prediction,sentence)
        return out_str

    '''
    >>> select the sentences
    '''
    def select(self, in_file):
        top_sentence_list=self.model.do_summarization(file_list=[in_file,],folder2store=self.folder2store,
            data_generator=self.generator,n_top=self.n_top)[0]
        if sys.version_info.major==2:
            top_sentence_list=sorted(top_sentence_list,lambda x,y: -1 if x[0]<y[0] else 1)
        else:
            top_sentence_list=sorted(top_sentence_list,key=lambda x:x[0],reverse=False)
        out_str=''
        for sentence_idx,prediction,sentence in top_sentence_list:
            out_str+='%s\n'%sentence
        return out_str

    def __del__(self):
        if os.path.exists(self.folder2store):
            shutil.rmtree(self.folder2store)

    '''
    >>> load dictionary
    '''
    def __load_dict__(self, file2load, format):
        if not format.lower() in ['txt','pkl']:
            raise ValueError('Unrecognized format: %s'%format)

        word2idx={}     # map<str -> (idx,entity_bit)>
        if format.lower() in ['pkl']:
            if sys.version_info.major==2:
                info=cPickle.load(open(file2load,'rb'))
            else:
                info=cPickle.load(open(file2load,'rb'),encoding='latin1')
            for word,frequency,global_idx,local_idx in info['word_list']:
                word2idx[word]=(local_idx,0)
            if 'entity_list' in info:
                for entity,frequency,global_idx,local_idx in info['entity_list']:
                    word2idx[entity]=(local_idx,1)

        if format.lower() in ['txt']:
            with open(file2load,'r') as fopen:
                header=fopen.readline().split(' ')
                word_list_length=int(header[0])
                entity_list_length=int(header[1]) if len(header)>1 else 0
                for word_idx in xrange(word_list_length):
                    parts=fpoen.readline().split(' ')
                    local_idx=int(parts[-1])
                    global_idx=int(parts[-2])
                    word=' '.join(parts[:-3])
                    assert(global_idx>0)
                    word2idx[word]=(local_idx,0)
                for entity_idx in xrange(entity_list_length):
                    parts=fopen.readline().split(' ')
                    local_idx=int(parts[-1])
                    global_idx=int(parts[-2])
                    entity=' '.join(parts[:-3])
                    assert(global_idx<0)
                    word2idx[entity]=(local_idx,1)

        return word2idx

if __name__=='__main__':

    if len(sys.argv)!=2:
        print('Usage: python laucher.py <config>')
        exit(0)

    hyper_params=xml_parser.parse(sys.argv[1],flat=False)
    my_launcher=laucher(hyper_params)

    while True:
        print('Type in a file to analyze, type bash/sh to launch bash, type exit to quit')
        answer=input('>>>')

        if answer.lower() in ['sh','bash']:
            os.system('bash')
        elif answer.lower() in ['exit']:
            break
        else:
            if not os.path.exists(answer):
                print('File not exists!')
            else:
                output=my_launcher.run(answer)
                cPickle.dump(output,open('temp.pkl','wb'))
                print(output)

