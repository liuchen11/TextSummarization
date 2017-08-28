import os
import sys
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
sys.path.insert(0,'util')
from py2py3 import *
import traceback

'''
>>> parse documens in dailymail corpus
>>> return the following information: url, document sentences, the labels for these sentences, hightlights, entity2name mapping
'''
def parse_document(file_name,replace_entity=True):
    try:
        contents=open(file_name,'r').readlines()
        contents=''.join(contents)
        parts=contents.split('\n\n')
        if len(parts)<4:
            print('invalid file format in file: %s'%file_name)
            return None
        elif len(parts)>4:
            print('weired file format in file: %s'%file_name)
            print('this file has %d parts'%len(parts))
            return None
        url,sentence_label,highlights,entity_map=parts[:4]

        url=map(lambda x: x[:-1] if x[-1]=='\n' else x, url.split('\n'))

        def split_entity_name(line):
            if line in ['',None]:
                return [None,None]
            segments=line.split(':') if line[-1]!='\n' else line[:-1].split(':')
            return [segments[0],':'.join(segments[1:])]

        entity_map=map(split_entity_name, entity_map.split('\n'))
        entity2name={}
        entity_map=sorted(entity_map,lambda x,y: 1 if len(x)<len(y) else -1)
        for entity,name in entity_map:
            if entity==None:
                continue
            entity2name[entity]=name
            if replace_entity:
                sentence_label=sentence_label.replace(entity,name)
                highlights=highlights.replace(entity,name)

        sentence_label=map(lambda x: x[:-1] if x[-1]=='\n' else x, sentence_label.split('\n'))
        sentences=map(lambda x: x.split('\t')[0], sentence_label)
        labels=map(lambda x: int(x.split('\t')[-1]), sentence_label)

        highlights=map(lambda x: x[:-1] if x[-1]=='\n' else x, highlights.split('\n'))

        return {'url':url, 'sentences':sentences, 'labels':labels, 'highlights':highlights, 'entity2name':entity2name}
    except:
        traceback.print_exc()
        raise Exception('Error occurs when parsing file: %s'%file_name)

'''
>>> get the raw text file and saved
>>> write content to the output file or terminal if output_file is None
'''
def get_raw_text(input_file, output_file):
    struct_info=parse_document(input_file)
    if output_file==None:
        for idx,sentence in enumerate(struct_info['sentences']):
            print('%d\t%s'%(idx,sentence))
    else:
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        with open(output_file,'w') as fopen:
            for idx,sentence in enumerate(struct_info['sentences']):
                fopen.write('%s\n'%sentence)

'''
>>> batch version of get_raw_text
'''
def get_raw_text_folder(input_folder, output_folder, recursive_flag=False):
    file_pairs=[]
    if recursive_flag==False:
        for file in os.listdir(input_folder):
            if os.path.isfile(input_folder+os.sep+file) and file.split('.')[-1] in ['summary',]:
                input_file=input_folder+os.sep+file
                output_file=output_folder+os.sep+file
                file_pairs.append([input_file,output_file])
    else:
        for subdir,dirs,files in os.walk(input_folder):
            for file in files:
                if file.split('.')[-1] in ['summary',]:
                    input_file=subdir+os.sep+file
                    output_file=subdir.replace(input_folder,output_folder)+os.sep+file
                    file_pairs.append([input_file,output_file])

    for idx,(input_file,output_file) in enumerate(file_pairs):
        sys.stdout.write('Processing %d/%d - %.1f%%\r'%(idx+1,len(file_pairs),float(idx+1)/float(len(file_pairs))*100))
        get_raw_text(input_file, output_file)

