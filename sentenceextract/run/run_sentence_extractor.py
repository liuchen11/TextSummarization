import os
import sys
import numpy as np
import tensorflow as tf

if sys.version_info.major==2:
    import cPickle
else:
    import pickle as cPickle
    xrange=range

sys.path.insert(0, './model')
sys.path.insert(0, './util')

from py2py3 import *
import loader
import xml_parser
import data_loader
import data_generator
import embedding_loader
import sentence_extractor

if len(sys.argv)!=2:
    print('Usage: python run_sentence_extractor.py <config>')
    exit(0)

hyper_params=xml_parser.parse(sys.argv[1], flat=False)

# Build or load the word list (and entity list)
loader_params=hyper_params['loader']
data_loader_params=loader_params['data_loader']
src_folder_list2build_list=loader_params['src_folder_list2build_list']
dest_folder_list2build_list=loader_params['dest_folder_list2build_list'] if 'dest_folder_list2build_list' in loader_params else None
src_folder_list2parse=loader_params['src_folder_list2parse']
dest_folder_list2parse=loader_params['dest_folder_list2parse']
list_saved_format=loader_params['list_saved_format']

my_data_loader=data_loader.data_loader(data_loader_params)
# my_data_loader.build_lists(src_folder_list2build_list,dest_folder_list2build_list,list_saved_format)
# my_data_loader.build_idx_files(src_folder_list2parse,dest_folder_list2parse)
my_data_loader.load_dict()

# Construct the data_generator
generator_params=hyper_params['generator_params']
data_generator_params=generator_params['data_generator_params']

my_data_generator=data_generator.data_generator(data_generator_params)
my_data_generator.load(word_file=my_data_loader.word_list_file, entity_file=my_data_loader.entity_list_file,
    format=loader_params['list_saved_format'])

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
input_extend_tags=network_params['input_extend_tags'] if 'input_extend_tags' in network_params else []
model2load=network_params['model2load']

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
assert my_network.batch_size==1, 'The batch size of the network must be 1'
my_network.train_validate_test_init()
my_network.load_params(model2load)

folder2scan_list=hyper_params['folder2scan_list']
result_file=hyper_params['result_file']

# Scan the folder
file_list=[]
for folder2scan in folder2scan_list:
    for file in os.listdir(folder2scan):
        if file.split('.')[-1] in ['info',]:
            file_list.append(folder2scan+os.sep+file)
print('Detected %d documents in total'%len(file_list))

# Processing the document one by one
to_save=[]
for idx, file_name in enumerate(file_list):
    sys.stdout.write('Loading files: %d/%d=%.1f%%\r'%(idx+1, len(file_list), float(idx+1)/float(len(file_list))*100))
    sys.stdout.flush()
    my_data_generator.init_batch_gen(set_label=file_name, file_list=[file_name,], permutation=False)

    result_this_document={'file_name':file_name, 'result':[], 'ground_truth':[]}

    input_matrix, masks, labels, stop, extend_part=my_data_generator.batch_gen(set_label=file_name, batch_size=1,
        label_policy='min', extend_tags=input_extend_tags, model_tag='sentence_extract')
    assert stop==True
    predictions=my_network.test(inputs=input_matrix, masks=masks, extend_part=extend_part, fine_tune=True)[0]       # of size [document_length_threshold]
    masks=masks[0]
    labels=labels[0]

    for idx, (prediction, mask, label) in enumerate(zip(predictions, masks, labels)):
        if mask==1:
            result_this_document['result'].append((idx, prediction))
            result_this_document['ground_truth'].append(label)

    if sys.version_info.major==2:
        result_this_document['result']=sorted(result_this_document['result'], lambda x,y: 1 if x[1]<y[1] else -1)
    else:
        result_this_document['result']=sorted(result_this_document, key=lambda x:x[1], reverse=True)

    to_save.append(result_this_document)

cPickle.dump(to_save, open(result_file, 'wb'))

