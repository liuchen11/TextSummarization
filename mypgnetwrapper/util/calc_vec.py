import os
import sys
import traceback
import numpy as np
if sys.version_info.major==2:
    import cPickle as pickle
else:
    import pickle

sys.path.insert(0, './util')
sys.path.insert(0, './model')
import convnet
import xml_parser

if len(sys.argv)<2:
    print('Usage: python rouge.py <config>')
    exit(0)

hyper_param=xml_parser.parse(sys.argv[1])

saved_file=hyper_param['saved_file']
refer_folder=hyper_param['refer_folder']
output_folder=hyper_param['output_folder']
refer_suffix=hyper_param['refer_suffix'] if 'refer_suffix' in hyper_param else 'reference'
output_suffix=hyper_param['output_suffix'] if 'output_suffix' in hyper_param else 'decode'

convnet_params=hyper_param['convnet_params']
model2load=hyper_param['model2load']
word2idx_file=hyper_param['word2idx_file']
unk_idx=hyper_param['unk_idx']

# Collect files
refer_name2file={}
output_name2file={}

allowed_format=['txt',]
for refer_file in os.listdir(refer_folder):
    if refer_file.split('.')[-1] in allowed_format:
        pure_name='.'.join(refer_file.split('.')[:-1])
        mark=pure_name.split('_')[0]
        suffix=pure_name.split('_')[-1]
        if suffix==refer_suffix:
            refer_name2file[mark]=refer_folder+os.sep+refer_file
print('In reference folder, there are %d detected files'%len(refer_name2file.keys()))
for output_file in os.listdir(output_folder):
    if output_file.split('.')[-1] in allowed_format:
        pure_name='.'.join(output_file.split('.')[:-1])
        mark=pure_name.split('_')[0]
        suffix=pure_name.split('_')[-1]
        if suffix==output_suffix:
            output_name2file[mark]=output_folder+os.sep+output_file
print('In the decode folder, there are %d detected files'%len(output_name2file.keys()))

name2files={}
for key in refer_name2file:
    if key in output_name2file:
        name2files[key]=(refer_name2file[key],output_name2file[key])
print('There are %d reference-decode paire detected'%len(name2files.keys()))

# Load model
my_convnet=convnet.convnet(convnet_params)
my_convnet.train_validate_test_init()
my_convnet.load_params(file2dump=model2load)

word2idx=cPickle.load(open(word2idx_file, 'r'))

tosave={'summary':{'cosine_average':0.0, 'dist_average':0.0, 'valid_documents':0},'details':[]}
r=rouge.Rouge()

for idx,key in enumerate(name2files):
    sys.stdout.write('loading documents %d/%d=%.1f%%\r'%((idx+1),len(name2files.keys()),
        float(idx+1)/float(len(name2files.keys()))*100))
    sys.stdout.flush()
    refer_file,output_file=name2files[key]
    try:
        refer_sequence=open(refer_file, 'r').readlines()
        output_sequence=open(output_file, 'r').readlines()
        refer_sequence=map(lambda x: x if x[-1]!='\n' else x[:-1], refer_sequence)
        output_sequence=map(lambda x: x if x[-1]!='\n' else x[:-1], output_sequence)
        refer_sequence=' '.join(refer_sequence)
        output_sequence=' '.join(output_sequence)

        refer_vec=my_convnet.sequence2vec(input_text=refer_sequence, word2idx=word2idx, unknown_idx=unk_idx)
        output_vec=my_convnet.sequence2vec(input_text=output_sequence, word2idx=word2idx, unknown_idx=unk_idx)

        cosine_value=np.dot(refer_vec, output_vec)/np.linalg.norm(refer_vec, 2)/np.linalg.norm(output_vec, 2)
        dist_value=np.linalg.norm(np.array(refer_vec)-np.array(output_vec), 2)
    except:
        print('ERROR!! refer_file: %s, decode_file: %s'%(refer_file, output_file))
        traceback.print_exc()
        continue
    tosave['details'].append({'cosine_value':cosine_value, 'dist_value':dist_value})
    tosave['summary']['cosine_average']+=cosine_value
    tosave['summary']['dist_average']+=dist_value
    tosave['summary']['valid_documents']+=1

    if tosave['summary']['valid_documents']%500==0:
        print('Rouge score of first %s documents has been saved in %s'%(
            tosave['summary']['valid_documents'],saved_file))
        pickle.dump(tosave,open(saved_file,'wb'))

print('All documents are loaded and processing completely!')
if tosave['summary']['valid_documents']!=0:
    tosave['summary']['cosine_average']/=tosave['summary']['valid_documents']
    tosave['summary']['dist_average']/=tosave['summary']['valid_documents']

pickle.dump(tosave,open(saved_file,'wb'))

