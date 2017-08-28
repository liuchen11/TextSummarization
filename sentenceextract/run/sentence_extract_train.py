import os
import sys
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
    import pickle as cPickle
else:
    import cPickle
sys.path.insert(0,'./util')
from py2py3 import *
import shutil
import numpy as np

sys.path.insert(0,'util')
sys.path.insert(0,'model')

import data_loader
import data_generator
import embedding_loader
import sentence_extractor
import xml_parser

if len(sys.argv)!=2:
    print('Usage: python sentence_extract_train.py <config>')
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

my_data_loader=data_loader.data_loader(data_loader_params)
my_data_loader.build_lists(src_folder_list2build_list,dest_folder_list2build_list,list_saved_format)
# my_data_loader.load_dict()
my_data_loader.build_idx_files(src_folder_list2parse,dest_folder_list2parse)

# Construct the data_generator
generator_params=hyper_params['generator_params']
data_generator_params=generator_params['data_generator_params']
data_sets=generator_params['data_sets']

my_data_generator=data_generator.data_generator(data_generator_params)
my_data_generator.load(word_file=my_data_loader.word_list_file, entity_file=my_data_loader.entity_list_file, format=loader_params['list_saved_format'])
for key in data_sets:
    my_data_generator.init_batch_gen(set_label=key,file_list=data_sets[key],permutation=True)

# Process word embedding
embedding_params=hyper_params['embedding_params']
embedding_loader_params=embedding_params['embedding_loader_params']
source=embedding_params['source']
format=embedding_params['format']
my_embedding_loader=embedding_loader.embedding_loader(embedding_loader_params)
my_embedding_loader.load_embedding(source=source,format=format)
embedding_matrix=my_embedding_loader.gen_embedding_matrix(generator=my_data_generator)

# Construct the neural network
network_params=hyper_params['network_params']
se_model_params=network_params['se_model_params']
model_saved_folder=network_params['model_saved_folder']
batch_num=network_params['batches']
validation_frequency=network_params['validation_frequency']
validation_batches=network_params['validation_batches']
input_extend_tags=network_params['input_extend_tags']
gpu_memory_ratio=0.9 if not 'gpu_memory_ratio' in network_params else network_params['gpu_memory_ratio']
check_err_frequency=max(1,validation_frequency/10)
begin_batch_idx=0 if not 'begin_batch_idx' in network_params else network_params['begin_batch_idx']

assert(se_model_params['sequence_length']==my_data_generator.sentence_length_threshold)
assert(se_model_params['sequence_num']==my_data_generator.document_length_threshold)
if my_data_generator.enable_entity_bit==False:
    assert(se_model_params['vocab_size']==my_data_generator.word_list_length+2)
else:
    assert(se_model_params['vocab_size']==my_data_generator.word_list_length+my_data_generator.entity_list_length+3)
assert(se_model_params['embedding_dim']==my_embedding_loader.embedding_dim)
if 'pretrain_embedding' in network_params and network_params['pretrain_embedding']==True:
    se_model_params['embedding_matrix']=embedding_matrix

my_network=sentence_extractor.sentence_extractor(se_model_params)
if not os.path.exists(model_saved_folder):
    os.makedirs(model_saved_folder)

training_loss=[]
my_network.train_validate_test_init(gpu_memory_ratio)       # initialization of training
if 'model2load' in network_params:
    model2load_file=network_params['model2load']
    my_network.load_params(model2load_file)
best_validation_loss=1e8                    # best validation loss
best_pt=-1                                  # best point in validation

train_loss_dict={}
validate_loss_dict={}
if os.path.exists(model_saved_folder+os.sep+'%s_info.pkl'%my_network.name):
    information=cPickle.load(open(model_saved_folder+os.sep+'%s_info.pkl'%my_network.name,'rb'))
    print('Information loaded from %s'%(model_saved_folder+os.sep+'%s_info.pkl'%my_network.name))
    train_loss_dict=information['train']
    validate_loss_dict=information['validate']

for batch_idx in xrange(begin_batch_idx,batch_num+begin_batch_idx):
    input_matrix,masks,labels,_,extension_part=my_data_generator.batch_gen(set_label='train',batch_size=my_network.batch_size,extend_tags=input_extend_tags,label_policy='min')
    ratio=min(1.0, batch_idx/10000)
    _, loss=my_network.train(input_matrix,masks,labels,ratio,extension_part)
    sys.stdout.write('Batch_idx: %d/%d, loss=%.4f\r'%(batch_idx+1,batch_num+begin_batch_idx,loss))
    training_loss.append(loss)

    if (batch_idx+1)%check_err_frequency==0:        # plot the loss average
        print('Average loss in [%d,%d)=%.4f'%(batch_idx+1-check_err_frequency,batch_idx+1,np.mean(training_loss[-check_err_frequency:])))
        train_loss_dict[batch_idx+1]=np.mean(training_loss[-check_err_frequency:])

    if (batch_idx+1)%validation_frequency==0:       # start validation
        my_network.dump_params(file2dump=model_saved_folder+os.sep+'%s_%d.ckpt'%(my_network.name,batch_idx+1))
        validation_loss=[]
        my_data_generator.init_batch_gen(set_label='validate',file_list=None,permutation=True)        # Permutation is done before each validation
        for validation_batch_idx in xrange(validation_batches):
            input_matrix,masks,labels,_,extension_part=my_data_generator.batch_gen(set_label='validate',batch_size=my_network.batch_size,extend_tags=input_extend_tags,label_policy='min')
            _, loss=my_network.validate(input_matrix,masks,labels,1.0,extension_part)
            validation_loss.append(loss)
            sys.stdout.write('Validation Batch_idx %d/%d, loss=%.4f, average=%.4f\r'%(validation_batch_idx,validation_batches,loss,np.mean(validation_loss)))
        if np.mean(validation_loss)<best_validation_loss:
            best_validation_loss=np.mean(validation_loss)
            best_pt=batch_idx+1
        validate_loss_dict[batch_idx+1]=np.mean(validation_loss)
        print('')

my_network.train_validate_test_end()
print('Best validation model: %s'%(model_saved_folder+os.sep+'%s_%d.ckpt'%(my_network.name,best_pt)))
cPickle.dump({'train':train_loss_dict,'validate':validate_loss_dict},open(model_saved_folder+os.sep+'%s_info.pkl'%my_network.name,'wb'))
print('Training information saved in %s_info.pkl'%my_network.name)

