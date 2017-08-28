import os
import sys
import random

import tensorflow as tf
import numpy as np
import traceback

import glob
import struct
from tensorflow.core.example import example_pb2

file_path=os.path.dirname(os.path.realpath(__file__))
file_path=os.path.join(os.path.join(file_path,os.pardir),'model')
sys.path.insert(0, file_path)
import util
import html
import data
import model
import batcher
import beam_search

SENTENCE_START='<s>'
SENTENCE_FINISH='</s>'


class BinaryDataManager(object):

    def __init__(self,binary_file,single_pass):
        if not os.path.exists(binary_file):
            raise ValueError('File not exists: %s'%binary_file)
        self.binary_file=binary_file

        if len(self.binary_file)==0:
            raise ValueError('No valid file')
        
        self.single_pass=single_pass
        self.text_abstract_pair=[]

    def __example_generator__(self,):
        file_list=glob.glob(self.binary_file)
        if self.single_pass:
            file_list=sorted(file_list)
        else:
            file_list=random.shuffle(file_list)

        while True:
            for file in file_list:
                fopen=open(file,'rb')
                while True:
                    len_bytes=fopen.read(8)
                    if not len_bytes:
                        break
                    str_len=struct.unpack('q', len_bytes)[0]
                    example_str=struct.unpack('%ds' % str_len, fopen.read(str_len))[0]
                    example=example_pb2.Example.FromString(example_str)
                    try:
                        article=example.features.feature['article'].bytes_list.value[0]
                        abstract=example.features.feature['abstract'].bytes_list.value[0]
                        if len(abstract)>0 and len(article)>0:
                            yield (article, abstract)
                    except ValueError:
                        print('Failed to get the abstract or article from example: %s'%str(example))
                        continue

            if self.single_pass:
                break

    def __clean_abstract__(self,abstract):
        cur=0
        sentence_list=[]
        while True:
            try:
                start_index=abstract.index(SENTENCE_START,cur)
                end_index=abstract.index(SENTENCE_FINISH,start_index+1)
                cur=end_index+len(SENTENCE_FINISH)
                sentence_list.append(abstract[start_index+len(SENTENCE_START):end_index])
            except:
                break
        return sentence_list

    def load_data(self,):
        generator=self.__example_generator__()

        while True:

            try:
                article, abstract=generator.next()
            except StopIteration:
                if self.single_pass==True:
                    print('Data loading is completed! There are %d pairs in total'%(len(self.text_abstract_pair)))
                    break
                else:
                    raise ValueError('Data is exhausted, but the single pass flag is not set True')

            abstract=self.__clean_abstract__(abstract)
            self.text_abstract_pair.append((article,abstract))
            sys.stdout.write('%d pairs are already loaded!\r'%len(self.text_abstract_pair))



