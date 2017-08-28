import os
import sys
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
sys.path.insert(0,'./util')
from py2py3 import *
import tensorflow as tf
import numpy as np

sys.path.insert(0,'./model')
sys.path.insert(0,'./util')

import data_loader
import data_generator
import embedding_loader
import sentence_extractor
import xml_parser

if len(sys.argv)<2:
    print('Usage: python sentence_extract_test.py <config> [<reload>]')
    exit(0)

hyper_params=xml_parser.parse(sys.argv[1],flat=False)

# Build word list or entity list
loader_params=hyper_params['loader']
data_loader_params=loader_params['data_loader']
src_folder_list2build_list=loader_params['src_folder_list2build_list']
dest_folder_list2build_list=loader_params['dest_folder_list2build_list'] if 'dest_folder_list2build_list' in loader_params else None
src_folder_list2parse=loader_params['src_folder_list2parse']
dest_folder_list2parse=loader_params['dest_folder_list2parse']
list_saved_format=loader_params['list_saved_format']

my_data_loader=data_loader.data_loader(data_loader_params)
if sys.argv[2].lower() in ['reload','r','-r','--r','-reload','--reload']:
    print('Rebuild the dictionary')
    my_data_loader.build_lists(src_folder_list2build_list,dest_folder_list2build_list,list_saved_format)
    my_data_loader.build_idx_files(src_folder_list2parse,dest_folder_list2parse)
else:
    if len(sys.argv)>=3:
        print('ignore unrecognized flag %s'%sys.argv[2])
    my_data_loader.load_dict()

# Construct the data_generator
generator_params=hyper_params['generator_params']
data_generator_params=generator_params['data_generator_params']
data_sets=generator_params['data_sets']

my_data_generator=data_generator.data_generator(data_generator_params)
my_data_generator.load(word_file=my_data_loader.word_list_file, entity_file=my_data_loader.entity_list_file, format=loader_params['list_saved_format'])
for key in data_sets:
    my_data_generator.init_batch_gen(set_label=key,file_list=data_sets[key],permutation=True)

# Process word embedding
# embedding_params=hyper_params['embedding_params']
# embedding_loader_params=embedding_params['embedding_loader_params']
# source=embedding_params['source']
# format=embedding_params['format']
# my_embedding_loader=embedding_loader.embedding_loader(embedding_loader_params)
# my_embedding_loader.load_embedding(source=source,format=format)
# embedding_matrix=my_embedding_loader.gen_embedding_matrix(generator=my_data_generator)

# Construct the neural network
network_params=hyper_params['network_params']
se_model_params=network_params['se_model_params']
input_extend_tags=network_params['input_extend_tags'] if 'input_extend_tags' in network_params else []
model2load=network_params['model2load']
gpu_memory_ratio=network_params['gpu_memory_ratio'] if 'gpu_memory_ratio' in network_params else 0.75

assert(se_model_params['sequence_length']==my_data_generator.sentence_length_threshold)
assert(se_model_params['sequence_num']==my_data_generator.document_length_threshold)
if my_data_generator.enable_entity_bit==False:
    assert(se_model_params['vocab_size']==my_data_generator.word_list_length+2)
else:
    assert(se_model_params['vocab_size']==my_data_generator.word_list_length+my_data_generator.entity_list_length+3)
# assert(se_model_params['embedding_dim']==my_embedding_loader.embedding_dim)
# if 'pretrain_embedding' in network_params and network_params['pretrain_embedding']==True:
#    se_model_params['embedding_matrix']=embedding_matrix

my_network=sentence_extractor.sentence_extractor(se_model_params)

test_case_num=0
test_right_num=0
positive_num=0
negative_num=0
my_network.train_validate_test_init(gpu_memory_ratio)
my_network.load_params(model2load)
while True:
    input_matrix,masks,labels,stop,extend_part=my_data_generator.batch_gen(
        set_label='test',batch_size=my_network.batch_size,label_policy='min',extend_tags=input_extend_tags)

    predictions=my_network.test(inputs=input_matrix,masks=masks,extend_part=extend_part)
    masks=np.array(masks).reshape(-1)
    labels=np.array(labels).reshape(-1)
    predictions=np.array(predictions).reshape(-1)
    hits=map(lambda x: x[2] if x[0]==x[1] else 0, zip(labels,predictions,masks))
    positive_bit=map(lambda x: 1 if x[0]==1 and x[1]==1 else 0, zip(labels,masks))
    negative_bit=map(lambda x: 1 if x[0]==0 and x[1]==1 else 0, zip(labels,masks))
    positive_num+=np.sum(positive_bit)
    negative_num+=np.sum(negative_bit)
    test_case_num+=np.sum(masks)
    test_right_num+=np.sum(hits)
    sys.stdout.write('test_accuracy=%d/%d=%.1f%%, positive=%d(%.1f%%), negative=%d(%.1f%%)\r'%(test_right_num,test_case_num,float(test_right_num)/float(test_case_num)*100,
        positive_num,float(positive_num)/float(positive_num+negative_num)*100,negative_num,float(negative_num)/float(positive_num+negative_num)*100))

    if stop==True:
        break

print('')
