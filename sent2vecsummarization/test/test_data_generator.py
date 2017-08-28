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
import xml_parser
import numpy as np

import data_loader
import data_generator

if len(sys.argv)!=2:
    print('Usage: python test_data_generator.py <config>')
    exit(0)

hyper_params=xml_parser.parse(sys.argv[1],flat=False)
data_loader_params=hyper_params['data_loader_params']
data_generator_params=hyper_params['data_generator_params']

word_file=hyper_params['word_file']
entity_file=hyper_params['entity_file']
idx2idx_file=hyper_params['idx2idx_file']
batch_size=hyper_params['batch_size']
data_sets=hyper_params['data_sets']

my_data_loader=data_loader.data_loader(data_loader_params)
my_data_generator=data_generator.data_generator(data_generator_params)

my_data_loader.load_dict()
my_data_generator.load(word_file=word_file, entity_file=entity_file)
my_data_generator.dump_idx2idx(file2dump=idx2idx_file)

idx_local2global=my_data_generator.get_idx_local2global()
# cPickle.dump(idx_local2global,open('idx_local2global.pkl','wb'))
# cPickle.dump(my_data_generator.idx_global2local,open('idx_global2local.pkl','wb'))
idx_local2word={}
for local_idx in idx_local2global:
    global_idx=idx_local2global[local_idx]
    if global_idx>0:
        word=my_data_loader.word_list[global_idx-1]
    else:
        word=my_data_loader.entity_list[-global_idx-1]
    idx_local2word[local_idx]=word

for key in data_sets:
    my_data_generator.init_batch_gen(set_label=key, file_list=data_sets[key], permutation=True)

results=my_data_generator.batch_gen(set_label=data_sets.keys()[0], batch_size=batch_size, label_policy='min', extend_tags=[], model_tag='abstractive')

for instance_idx in xrange(batch_size):
    file_name=results['file_list'][instance_idx]
    encode_input=results['encode_input_batch'][instance_idx]
    encode_length=results['encode_input_length'][instance_idx]
    decode_input=results['decode_input_batch'][instance_idx]
    decode_refer=results['decode_refer_batch'][instance_idx]
    decode_mask=results['decode_mask'][instance_idx]

    encode_content=map(lambda x: idx_local2word[x], encode_input)
    decode_content=map(lambda x: idx_local2word[x], decode_input)
    refer_content=map(lambda x: idx_local2word[x], decode_refer)

    print('======================')
    print('file: %s'%file_name)
    print('encode content: \n %s'%(' '.join(encode_content)))
    print('encode length: %d'%encode_length)
    print('decode content: \n %s'%(' '.join(decode_content)))
    print('refer content: \n %s'%(' '.join(refer_content)))
    print('decode mask: %s'%decode_mask)
