import os
import sys
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
sys.path.insert(0,'./util')
from py2py3 import *
import shutil
import numpy as np
import cPickle

sys.path.insert(0,'util')
sys.path.insert(0,'model')

import network
import data_manager
import embedding_manager
import xml_parser

if len(sys.argv)!=2:
    print('Usage: python sentence_extract.py <config>')
    exit(0)

hyper_params=xml_parser.parse(file=sys.argv[1],flat=False)

# Process dataset
data_process_params=hyper_params['data_process']
force_flag=data_process_params['force']
data_manager_params=data_process_params['data_manager_params']
file_sets=data_process_params['file_sets']

my_data_manager=data_manager.data_manager(data_manager_params)
#my_data_manager.analyze_documents()
#my_data_manager.build_files(force=force_flag)
my_data_manager.load_dict()
for key in file_sets:
    file_set=file_sets[key]
    my_data_manager.init_batch_gen(set_label=key,file_list=file_set,permutation=True)

# Process word embedding
embedding_params=hyper_params['embedding']
embedding_manager_params=embedding_params['embedding_manager']
source=embedding_params['source']
format=embedding_params['format']
force=embedding_params['force']

my_embedding_manager=embedding_manager.embedding_manager(embedding_manager_params)
my_embedding_manager.load_embedding(source=source,format=format,force=force)
embedding_matrix=my_embedding_manager.gen_embedding_matrix(my_data_manager)

# Constructing the neural network
network_params=hyper_params['network']
sentence_extract_model_params=network_params['sentence_extract_model']
model_saved_folder=network_params['model_saved_folder']
batch_num=network_params['batches']
validation_frequency=network_params['validation_frequency']
validation_batches=network_params['validation_batches']
check_err_frequency=max(1,validation_frequency/10)
begin_batch_idx=0 if not 'begin_batch_idx' in network_params else network_params['begin_batch_idx']


assert(sentence_extract_model_params['sequence_length']==my_data_manager.sentence_length_threshold)
assert(sentence_extract_model_params['sequence_num']==my_data_manager.document_length_threshold)
assert(sentence_extract_model_params['vocab_size']==my_data_manager.word_list_length)
assert(sentence_extract_model_params['embedding_dim']==my_embedding_manager.embedding_dim+my_data_manager.extended_bits)
if 'pretrain_embedding' in network_params and network_params['pretrain_embedding']==True:
    sentence_extract_model_params['embedding_matrix']=embedding_matrix

my_network=network.sentenceExtractorModel(sentence_extract_model_params)
if not os.path.exists(model_saved_folder):
    os.makedirs(model_saved_folder)

training_loss=[]
my_network.train_validate_test_init()       # initialization of training
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
    input_matrix,masks,labels,_=my_data_manager.batch_gen(set_label='train',batch_size=my_network.batch_size,label_policy='min')
    ratio=min(1.0, batch_idx/10000)
    _, loss=my_network.train(input_matrix,masks,labels,ratio)
    sys.stdout.write('Batch_idx: %d/%d, loss=%.4f\r'%(batch_idx+1,batch_num+begin_batch_idx,loss))
    training_loss.append(loss)

    if (batch_idx+1)%check_err_frequency==0:        # plot the loss average
        print('Average loss in [%d,%d)=%.4f'%(batch_idx+1-check_err_frequency,batch_idx+1,np.mean(training_loss[-check_err_frequency:])))
        train_loss_dict[batch_idx+1]=np.mean(training_loss[-check_err_frequency:])

    if (batch_idx+1)%validation_frequency==0:       # start validation
        my_network.dump_params(file2dump=model_saved_folder+os.sep+'%s_%d.ckpt'%(my_network.name,batch_idx+1))
        validation_loss=[]
        my_data_manager.init_batch_gen(set_label='validate',file_list=None,permutation=True)        # Permutation is done before each validation
        for validation_batch_idx in xrange(validation_batches):
            input_matrix,masks,labels,_=my_data_manager.batch_gen(set_label='validate',batch_size=my_network.batch_size,label_policy='min')
            _, loss=my_network.validate(input_matrix,masks,labels,1.0)
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