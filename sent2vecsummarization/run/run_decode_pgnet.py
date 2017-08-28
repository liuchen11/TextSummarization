import os
import sys
sys.path.insert(0,'./util')
sys.path.insert(0,'./model')
import numpy as np
import tensorflow as tf

from py2py3 import *

import data_loader
import data_generator
import beam_search
import xml_parser

import pgnet

if len(sys.argv)!=2:
    print('Usage: python run_decode_pgnet.py <config>')
    exit(0)

hyper_params=xml_parser.parse(file=sys.argv[1], flat=False)

# Build word list or entity list
loader_params=hyper_params['loader']
data_loader_params=loader_params['data_loader']
src_folder_list2build_list=loader_params['src_folder_list2build_list']
dest_folder_list2build_list=loader_params['dest_folder_list2build_list'] if 'dest_folder_list2build_list' in loader_params else None
src_folder_list2parse=loader_params['src_folder_list2parse']
dest_folder_list2parse=loader_params['dest_folder_list2parse']
list_saved_format=loader_params['list_saved_format']

my_data_loader=data_loader.data_loader(data_loader_params)
if 'reload' in loader_params and loader_params['reload']==True:
    my_data_loader.build_lists(src_folder_list2build_list,dest_folder_list2build_list,list_saved_format)
    my_data_loader.build_idx_files(src_folder_list2parse,dest_folder_list2parse)
else:
    my_data_loader.load_dict()

# Construct the data_generator
generator_params=hyper_params['generator_params']
data_generator_params=generator_params['data_generator_params']
data_sets=generator_params['data_sets']

my_data_generator=data_generator.data_generator(data_generator_params)
my_data_generator.load(word_file=my_data_loader.word_list_file, entity_file=my_data_loader.entity_list_file, format=loader_params['list_saved_format'])
for key in data_sets:
    my_data_generator.init_batch_gen(set_label=key,file_list=data_sets[key],permutation=True)

# Construct the neural network
network_params=hyper_params['network_params']
pgnet_model_params=network_params['pgnet_model_params']
model2load=network_params['model2load']
gpu_ratio=0.25 if not 'gpu_ratio' in network_params else network_params['gpu_ratio']

assert pgnet_model_params['max_encoding_step']==my_data_generator.max_encoding_step, \
    'max_encoding_step of pgnet and generator do not match, pgnet=%d, generator=%d'%(
    pgnet_model_params['max_encoding_step'], my_data_generator.max_encoding_step)
# assert pgnet_model_params['max_decoding_step']==my_data_generator.max_decoding_step, \
#     'max_decoding_step of pgnet and generator do not match, pgnet=%d, generator=%d'%(
#     pgnet_model_params['max_decoding_step'], my_data_generator.max_decoding_step)

# Construct the beam search
search_params=hyper_params['search_params']
beam_search_params=search_params['beam_search_params']
output_text_folder=search_params['output_text_folder'] if 'output_text_folder' in  search_params else None
if output_text_folder!=None and not os.path.exists(output_text_folder):
    os.makedirs(output_text_folder)
max_processed_documents=search_params['max_processed_documents'] if 'max_processed_documents' in search_params else -1

my_beam_search_handler=beam_search.BeamSearchHandler(beam_search_params)

my_network=pgnet.PGNet(pgnet_model_params)
my_network.train_validate_test_init(gpu_prop=gpu_ratio)
my_network.load_params(model2load)

my_beam_search_handler.bind_model(my_network)
my_beam_search_handler.bind_generator(my_data_generator)
my_beam_search_handler.decode(set_label='test', file_list=None, fout=output_text_folder, single_pass=True, max_processed_documents=max_processed_documents)




