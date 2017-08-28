import os
import sys
import rouge
import traceback
if sys.version_info.major==2:
    import cPickle as pickle
else:
    import pickle

if len(sys.argv)<4:
    print('Usage: python rouge.py <output_file> <ground_truth folder> <output folder> [<ground_truth suffix>] [output suffix]')
    exit(0)

saved_file=sys.argv[1]
refer_folder=sys.argv[2]
output_folder=sys.argv[3]
refer_suffix=sys.argv[4] if len(sys.argv)>4 else 'reference'
output_suffix=sys.argv[5] if len(sys.argv)>5 else 'decode'

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

tosave={'summary':{'results':{'rouge-1':{'p':0.0,'r':0.0,'f':0.0},'rouge-2':{'p':0.0,'r':0.0,'f':0.0},
    'rouge-l':{'p':0.0,'r':0.0,'f':0.0}},'valid_documents':0},'details':[]}
r=rouge.Rouge()

for idx,key in enumerate(name2files):
    sys.stdout.write('loading documents %d/%d=%.1f%%\r'%((idx+1),len(name2files.keys()),
        float(idx+1)/float(len(name2files.keys()))*100))
    sys.stdout.flush()
    refer_file,output_file=name2files[key]
    try:
        refer_content=open(refer_file,'r').readlines()
        refer_content=' '.join(refer_content).decode('utf8').encode('ascii',errors='ignore')
        output_content=open(output_file,'r').readlines()
        output_content=' '.join(output_content).decode('utf8').encode('ascii',errors='ignore')
        single_result=r.get_scores(refer_content,output_content)[0]
    except:
        print('ERROR!! refer_file: %s, decode_file: %s'%(refer_file, output_file))
        traceback.print_exc()
        continue
    tosave['details'].append({'results':single_result,'reference':refer_file,'decode':output_file})
    for metric in ['p','r','f']:
        tosave['summary']['results']['rouge-1'][metric]+=single_result['rouge-1'][metric]
        tosave['summary']['results']['rouge-2'][metric]+=single_result['rouge-2'][metric]
        tosave['summary']['results']['rouge-l'][metric]+=single_result['rouge-l'][metric]
    tosave['summary']['valid_documents']+=1

    if tosave['summary']['valid_documents']%500==0:
        print('Rouge score of first %s documents has been saved in %s'%(
            tosave['summary']['valid_documents'],saved_file))
        pickle.dump(tosave,open(saved_file,'wb'))

print('All documents are loaded and processing completely!')
if tosave['summary']['valid_documents']!=0:
    for metric in ['p','r','f']:
        tosave['summary']['results']['rouge-1'][metric]/=tosave['summary']['valid_documents']
        tosave['summary']['results']['rouge-2'][metric]/=tosave['summary']['valid_documents']
        tosave['summary']['results']['rouge-l'][metric]/=tosave['summary']['valid_documents']
pickle.dump(tosave,open(saved_file,'wb'))





