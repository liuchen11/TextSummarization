import os
import sys
import time
import traceback
import numpy as np

if sys.version_info.major==2:
    input=raw_input

sys.path.insert(0, './model')
sys.path.insert(0, './util')

from py2py3 import *
import sent2vec_distributor

if len(sys.argv)<3:
    print('Usage: python test_sent2vec_distributor.py <fasttext_path> <fasttext_model> [<input_file>]')
    exit(0)

fasttext_path=sys.argv[1]
fasttext_model=sys.argv[2]
input_file=None if len(sys.argv)==3 else sys.argv[3]

if input_file!=None:
    assert os.path.exists(input_file), '%s does not exists'%input_file

assert os.path.exists(fasttext_model), '%s does not exists'%fasttext_model
assert os.path.exists(fasttext_path), '%s does not exists'%fasttext_path

start_time=time.time()
distributor=sent2vec_distributor.EmbeddingDistributor(fasttext_path=fasttext_path, fasttext_model=fasttext_model)
end_time=time.time()
print('Construction of the model takes %d seconds'%(end_time-start_time))

if input_file==None:
    print('Type in the sentence')
    while True:
        try:
            answer=input('>>>')
            start_time=time.time()
            embeddings=distributor.get_tokenized_sents_embeddings([answer,])
            end_time=time.time()
            print('The query takes %s seconds'%(end_time-start_time))
            print('Embeddings:')
            print('shape: %s'%(str(embeddings.shape)))
            print(embeddings)
        except:
            print('Interrupted!')
            traceback.print_exc()
            break
else:
    with open(input_file, 'r') as fopen:
        lines=fopen.readlines()
        lines=map(lambda x: x if x[-1]!='\n' else x[:-1], lines)
        start_time=time.time()
        embeddings=distributor.get_tokenized_sents_embeddings(lines)
        end_time=time.time()
        print('The query takes %s seconds, #line = %d'%(end_time-start_time, len(lines)))


