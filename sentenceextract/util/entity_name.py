import os
import sys
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
sys.path.insert(0,'util')
from py2py3 import *
import traceback

'''
>>> extract a entity-name dictionary and write it into a file
>>> folder_or_file_list: list<str>, list of files or folders
>>> saved_file: str, the document where we dump the dictionary
'''
def extract_entity_dict(folder_or_file_list, saved_file):
    file_list=[]
    for folder_or_file in folder_or_file_list:
        if os.path.isdir(folder_or_file):
            for file in os.listdir(folder_or_file):
                if file.split('.')[-1] in ['summary']:
                    file_list.append(folder_or_file+os.sep+file)
        elif os.path.isfile(folder_or_file):
            if folder_or_file.split('.')[-1] in ['summary']:
                file_list.append(folder_or_file)
        else:
            print('Invalid file or folder: %s'%folder_or_file)

    print('There are %d documents available'%(len(file_list)))

    entity2name={}
    for idx,file_name in enumerate(file_list):
        try:
            sys.stdout.write('loading documents %d/%d - %.1f%%\r'%(idx,len(file_list),float(idx)/float(len(file_list))*100))
            contents=open(file_name,'r').readlines()
            contents=''.join(contents)
            parts=contents.split('\n\n')
            entity_map=parts[-1]

            def split_entity_name(line):
                segments=line.split(':') if line[-1]!='\n' else line[:-1].split(':')
                return [segments[0],':'.join(segments[1:])]

            entity_map=map(split_entity_name, entity_map.split('\n'))
            for entity,name in entity_map:
                if entity in entity2name and entity2name[entity]!=name:
                    print('warning: key %s [%s -> %s]'%(entity, entity2name[entity], name))
                entity2name[entity]=name
        except:
            traceback.print_exc()
            print('Error occurs when parsing file: %s'%file_name)

    if saved_file!=None:
        with open(saved_file,'w') as fopen:
            for entity in entity2name:
                fopen.write('%s:%s\n'%(entity,entity2name[entity]))
    return entity2name

'''
>>> replace entity id with its real name
>>> entity2name: dict, id - real name mapping
>>> original_dict: str, file of original dict
>>> new_dict: str, file of new dict
'''
def replace_entity_name(entity2name, original_dict, new_dict):
    lines=open(original_dict,'r').readlines()
    lines=map(lambda x: x[:-1] if x[-1]=='\n' else x, lines)
    with open(new_dict,'w') as fopen:
        fopen.write(lines[0]+'\n')
        for idx,line in enumerate(lines[1:]):
            sys.stdout.write('scanned %d/%d words - %.1f%%\r'%(idx,len(lines),float(idx)/float(len(lines))*100))
            parts=line.split(' ')
            word=' '.join(parts[1:-1])
            if word in entity2name:
                name=entity2name[word]
                fopen.write('%s %s %s\n'(parts[0],name,parts[-1]))
            else:
                fopen.write('%s\n'%line)

if __name__=='__main__':

    if len(sys.argv)<4:
        print('entity_name.py <original_dict> <new_dict> <entity_saved_dict> <file_or_folder>...')
        exit(0)

    entity2name=extract_entity_dict(folder_or_file_list=sys.argv[4:], saved_file=sys.argv[3])
    replace_entity_name(entity2name=entity2name, original_dict=sys.argv[1], new_dict=sys.argv[2])
