import os
import sys
import cPickle
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
sys.path.insert(0,'./util')
from py2py3 import *

import xml_parser
import loader

if __name__=='__main__':

    if len(sys.argv)!=2:
        print('Usage: python process_entity_name.py <config>')
        exit(0)

    params=xml_parser.parse(sys.argv[1],flat=False)

    word_list_file=params['word_list_file']
    document_folder_list=params['document_folder_list']
    for folder in document_folder_list:
        assert(os.path.exists(folder))

    new_word_list_file=params['new_word_list_file']
    new_entity_list_file=params['new_entity_list_file']
    new_document_folder_list=params['new_document_folder_list']

    assert(len(document_folder_list)==len(new_document_folder_list))

    # Load word_list and entity_list
    word_list=[]
    word_frequency_list=[]
    word2idx={}
    with open(word_list_file,'r') as fopen:
        fopen.readline()
        for idx,line in enumerate(fopen):
            sys.stdout.write('%d words loaded\r'%(idx+1))
            parts=line.split(' ')
            word=' '.join(parts[1:-1])
            word=str(word.encode('utf8'))
            word2idx[word]=idx
            word_frequency_list.append(0)
            word_list.append(word)
    print('there are %d words in the dictionary in total'%len(word_frequency_list))

    entity_list=[]
    entity_frequency_list=[]
    entity2idx={}
    file_detected=0

    for src_folder,dest_folder in zip(document_folder_list,new_document_folder_list):
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        for file in os.listdir(src_folder):
            if file.split('.')[-1] in ['summary',]:
                file_detected+=1
                sys.stdout.write('%d files detected\r'%file_detected)
                document_info=loader.parse_document(src_folder+os.sep+file,replace_entity=False)
                dest_file=dest_folder+os.sep+file
                writer=open(dest_file,'w')
                writer.write(str(len(document_info['sentences']))+'\n')
                for sentence in document_info['sentences']:
                    sentence_idx_info=[]
                    sentence=sentence.split(' ')
                    for word in sentence:
                        word=str(word.encode('utf8'))
                        # idx_in_word_list=find_item_in_list(word_list,word)
                        if word in document_info['entity2name']:
                            entity_name=document_info['entity2name'][word]
                            # idx_in_entity_list=find_item_in_list(entity_list,entity_name)
                            if entity_name in entity2idx:
                                idx_in_entity_list=entity2idx[entity_name]
                                sentence_idx_info.append(-idx_in_entity_list)
                                entity_frequency_list[idx_in_entity_list]+=1
                            else:
                                # print('detected an unknown entity name in file %s: %s, added to entity_list'%(src_folder+os.sep+file,entity_name))
                                idx_in_entity_list=len(entity_list)
                                entity2idx[entity_name]=idx_in_entity_list
                                sentence_idx_info.append(-idx_in_entity_list)
                                entity_list.append(entity_name)
                                entity_frequency_list.append(1)
                        else:
                            if word in word2idx:
                                idx_in_word_list=word2idx[word]
                                sentence_idx_info.append(idx_in_word_list)
                                word_frequency_list[idx_in_word_list]+=1
                            else:
                                print('WARNING: unknown word in file %s: %s, marked as 1e10'%(src_folder+os.sep+file,word))
                                sentence_idx_info.append(1e10)
                    sentence_idx_info=map(str,sentence_idx_info)
                    writer.write(','.join(sentence_idx_info)+'\n')
                label_info=map(str,document_info['labels'])
                writer.write(','.join(label_info))
                writer.close()
    print('')
    cPickle.dump(zip(word_list,word_frequency_list),open(new_word_list_file,'wb'))
    cPickle.dump(zip(entity_list,entity_frequency_list),open(new_entity_list_file, 'wb'))
