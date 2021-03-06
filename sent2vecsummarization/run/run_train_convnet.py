import os
import sys
sys.path.insert(0,'./util')
sys.path.insert(0,'./model')
import numpy as np
import tensorflow as tf

from py2py3 import *
if sys.version_info.major==2:
    import cPickle
else:
    import pickle as cPickle

import data_loader
import data_generator
import embedding_loader
import xml_parser

import convnet

if len(sys.argv)!=2:
    print('Usage: python run_train_convnet.py <config>')
    exit(0)

hyper_params=xml_parser.parse(file=sys.argv[1],flat=False)

# Build word list or entity list
loader_params=hyper_params['loader']
data_loader_params=loader_params['data_loader']
src_folder_list2build_list=loader_params['src_folder_list2build_list']
dest_folder_list2build_list=loader_params['dest_folder_list2build_list'] if 'dest_folder_list2build_list' in loader_params else None
src_folder_list2parse=loader_params['src_folder_list2parse']
dest_folder_list2parse=loader_params['dest_folder_list2parse']
list_saved_format=loader_params['list_saved_format']
reload=loader_params['reload']

my_data_loader=data_loader.data_loader(data_loader_params)
my_data_loader.load_dict()
if reload==True:
    my_data_loader.build_lists(src_folder_list2build_list,dest_folder_list2build_list,list_saved_format,format='new_extractive',dictionary_fixed=True)
    my_data_loader.build_idx_files(src_folder_list2parse,dest_folder_list2parse, format='new_extractive')

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
convnet_model_params=network_params['convnet_model_params']
model_saved_folder=network_params['model_saved_folder']

batch_num=network_params['batch_num']
validation_frequency=network_params['validation_frequency']
validation_batches=network_params['validation_batches']
check_err_frequency=max(1, validation_frequency/10)
begin_batch_idx=0 if not 'begin_batch_idx' in network_params else network_params['begin_batch_idx']

if 'pretrained_embeddings' in network_params and network_params['pretrained_embeddings']==True:
    # Process word embedding
    embedding_params=hyper_params['embedding_params']
    embedding_loader_params=embedding_params['embedding_loader_params']
    source=embedding_params['source']
    format=embedding_params['format']
    my_embedding_loader=embedding_loader.embedding_loader(embedding_loader_params)

    assert convnet_model_params['embedding_dim']==my_embedding_loader.embedding_dim, \
        'dimensions of word embeddings do not match, convnet=%d, embedding_loader=%d'%(
        convnet_model_params['embedding_dim'], my_embedding_loader.embedding_dim)

    my_embedding_loader.load_embedding(source=source, format=format)
    embedding_matrix=my_embedding_loader.gen_embedding_matrix(generator=my_data_generator)
    convnet_model_params['embedding_matrix']=embedding_matrix

my_network=convnet.convnet(convnet_model_params)
if not os.path.exists(model_saved_folder):
    os.makedirs(model_saved_folder)

my_network.train_validate_test_init()
if 'model2load' in network_params:
    try:
        my_network.load_params(network_params['model2load'])
    except:
        print('ERROR: Failed to load checkpoint: %s'%network_params['model2load'])

train_loss_dict={}
validate_loss_dict={}
train_loss=[]
best_validation_loss=1e8                # best validation loss
best_pt=-1                              # best point in validation
if os.path.exists(model_saved_folder+os.sep+'%s_info.pkl'%my_network.name):
    try:
        information=cPickle.load(open(model_saved_folder+os.sep+'%s_info.pkl'%my_network.name, 'rb'))
        print('Information loaded from %s'%(model_saved_folder+os.sep+'%s_info.pkl'%my_network.name))
        train_loss_dict=information['train']
        validate_loss_dict=information['validate']
    except:
        print('Error occurs while loading file %s'%(model_saved_folder+os.sep+'%s_info.pkl'%my_network.name))

for batch_idx in xrange(begin_batch_idx, begin_batch_idx+batch_num):
    batch_info=my_data_generator.batch_gen(set_label='train', batch_size=my_network.batch_size, label_policy='min', model_tag='new_extractive')
    _, loss=my_network.train(inputs=batch_info['input_matrix'], labels=batch_info['labels'], extend_part=batch_info['extension_part'])
    sys.stdout.write('Batch_idx: %d/%d loss=%.4f\r'%(batch_idx+1, begin_batch_idx+batch_num, loss))
    sys.stdout.flush()
    train_loss.append(loss)

    if (batch_idx+1)%check_err_frequency==0:
        print('Average loss in [%d, %d)=%.4f\r'%(batch_idx+1-check_err_frequency, batch_idx, np.mean(train_loss)))
        train_loss_dict[batch_idx+1]=np.mean(train_loss)
        train_loss=[]

    if (batch_idx+1)%validation_frequency==0:
        my_network.dump_params(file2dump=model_saved_folder+os.sep+'%s_%d.ckpt'%(my_network.name, batch_idx+1))
        validation_loss=[]
        my_data_generator.init_batch_gen(set_label='validate', file_list=None, permutation=True)
        if validation_batches>0:
            for validation_batch_idx in xrange(validation_batches):
                batch_info=my_data_generator.batch_gen(set_label='validate', batch_size=my_network.batch_size, label_policy='min', model_tag='new_extractive')
                _, loss=my_network.validate(inputs=batch_info['input_matrix'], labels=batch_info['labels'], extend_part=batch_info['extension_part'])
                validation_loss.append(loss)
                sys.stdout.write('Validation Batch_idx=%d/%d, loss=%.4f, average=%.4f\r'%(validation_batch_idx+1, validation_batches, loss, np.mean(validation_loss)))
        else:
            while True:
                batch_info=my_data_generator.batch_gen(set_label='validate', batch_size=my_network.batch_size, label_policy='min', model_tag='new_extractive')
                _, loss=my_network.validate(inputs=batch_info['input_matrix'], labels=batch_info['labels'], extend_part=batch_info['extension_part'])
                validation_loss.append(loss)
                sys.stdout.write('Validation Batch_idx=%d, loss=%.4f, average=%.4f\r'%(len(validation_loss), loss, np.mean(validation_loss)))
                if batch_info['end_of_epoch']==True:
                    break

        if np.mean(validation_loss)<best_validation_loss:
            best_validation_loss=np.mean(validation_loss)
            best_pt=batch_idx+1
        validate_loss_dict[batch_idx+1]=np.mean(validation_loss)
        print('')

variable_list=tf.global_variables()
my_network.train_validate_test_end()
print('Best validation model: %s'%(model_saved_folder+os.sep+'%s_%d.ckpt'%(my_network.name, best_pt)))
cPickle.dump({'train':train_loss_dict, 'validate':validate_loss_dict, 'variable_list':variable_list},
    open(model_saved_folder+os.sep+'%s_info.pkl'%(my_network.name), 'wb'))
print('Training information saved in %s_info.pkl'%(my_network.name))

# feed_dict={my_network.encode_input_batch:batch_info['encode_input_batch'], my_network.encode_input_length:batch_info['encode_input_length'], my_network.decode_input_batch:batch_info['decode_input_batch'], my_network.decode_ground_truth:batch_info['decode_refer_batch'], my_network.decode_mask:batch_info['decode_mask'], my_network.init_decode_hidden_states:np.array([], dtype=np.float32).reshape([my_network.batch_size, 0, my_network.decoding_dim])}

