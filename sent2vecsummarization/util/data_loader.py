import os
import sys
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
    from builtins import input
    import pickle as cPickle
else:
    input=raw_input
    import cPickle
sys.path.insert(0,'util')
from py2py3 import *
import loader
import random
import numpy as np

'''
>>> data loader for abstractive models
'''
class data_loader(object):

    '''
    >>> Constructor
    '''
    def __init__(self, hyper_params):
        self.word_list=[]
        self.word_frequency=[]
        self.entity_list=[]
        self.entity_frequency=[]
        self.word2idx={}
        self.entity2idx={}

        self.enable_entity_detection=hyper_params['enable_entity_detection']
        self.word_list_file=hyper_params['word_list_file']
        self.entity_list_file=hyper_params['entity_list_file']

        # load special tokens
        self.word_list=['[START]', '[END]', '<s>', '</s>', '[UNK]', '[PAD]']
        self.word_frequency=[int(1e10), int(1e10), int(1e10), int(1e10), int(1e10), int(1e10)]
        for idx, word in enumerate(self.word_list):
            self.word2idx[word]=idx+1                   # idx for words begins with 1

    '''
    >>> load word/entity information into workspace
    '''
    def __load_list__(self, file2load, format):
        if not os.path.exists(file2load):
            raise Exception('File %s does not exists! Impossible to load it'%file2load)

        content=[]
        if format.lower() in ['txt',]:
            with open(file2load,'r') as fopen:
                for line in fopen:
                    line=line if line[-1]!='\n' else line[:-1]
                    parts=line.split(' ')
                    idx=int(parts[0])
                    word=' '.join(parts[1:-1])
                    frequency=int(parts[-1])
                    content.append((idx,word,frequency))
        elif format.lower() in ['pkl']:
            if sys.version_info.major==2:
                content=cPickle.load(open(file2load,'rb'))
            else:
                content=cPickle.load(open(file2load,'rb'),encoding='latin1')
        else:
            raise ValueError('Unrecognized list load format: %s'%format)
        return content

    '''
    >>> save the word/entity list into a file
    '''
    def __save_list__(self, content, file2dump, format):
        if os.path.exists(file2dump):
            print('File %s does exists, to overwrite it?'%file2dump)
            answer=input('Y/n >>> ')
            if answer.lower() in ['n','no']:
                return

        if format.lower() in ['txt',]:
            with open(file2dump,'w') as fopen:
                for idx,word,frequency in content:
                    fopen.write('%d %s %d\n'%(idx,word,frequency))
        elif format.lower() in ['pkl',]:
            cPickle.dump(content,open(file2dump,'wb'))
        else:
            raise ValueError('Unrecognized list saved format: %s'%format)

    '''
    >>> load dictionary
    '''
    def load_dict(self, list_load_format='pkl'):
        word_list_info=self.__load_list__(self.word_list_file, format=list_load_format)
        if sys.version_info.major==2:       # sort the list by idx
            word_list_info=sorted(word_list_info, lambda x,y: -1 if x[0]<y[0] else 1)
        else:
            word_list_info=sorted(word_list_info, key=lambda x:x[0], reverse=False)
        word_max_idx=word_list_info[-1][0]
        self.word_list=[None for idx in xrange(word_max_idx)]
        self.word_frequency=[None for idx in xrange(word_max_idx)]
        for idx,word,frequency in word_list_info:
            self.word_list[idx-1]=word
            self.word_frequency[idx-1]=frequency
            self.word2idx[word]=idx-1
        print('%d word loaded into the workspace'%(len(word_list_info)))

        if self.enable_entity_detection:
            entity_list_info=self.__load_list__(self.entity_list_file, format=list_load_format)
            if sys.version_info.major==2:
                entity_list_info=sorted(entity_list_info, lambda x,y: -1 if x[0]<y[0] else 1)
            else:
                entity_list_info=sorted(entity_list_info, key=lambda x:x[0], reverse=False)
            entity_max_idx=entity_list_info[-1][0]
            self.entity_list=[None for idx in xrange(entity_max_idx)]
            self.entity_frequency=[None for idx in xrange(entity_max_idx)]
            for idx,entity,frequency in entity_list_info:
                self.entity_list[idx-1]=entity
                self.entity_frequency[idx-1]=frequency
                self.entity2idx[entity]=idx-1
            print('%d entity name loaded into the workspace'%(
                len(entity_list_info)))


    '''
    >>> build word_list and entity_list
    >>> src_folder_list: list of source folders
    >>> dest_folder_list: list of destination folders, if None, idx of document will not be saved
    >>> list_saved_format: str in ['pkl','txt'], the format to save word or entity list
    '''
    def build_lists(self, src_folder_list, dest_folder_list, list_saved_format='pkl', format='abstractive', dictionary_fixed=False):
        if not list_saved_format.lower() in ['pkl','txt']:
            raise ValueError('Unrecognized list saved format: %s'%list_saved_format)

        if dest_folder_list==None:
            dest_folder_list=[None,]*len(src_folder_list)

        assert(len(src_folder_list)==len(dest_folder_list))

        file_list=[]
        for idx,(src_folder,dest_folder) in enumerate(zip(src_folder_list,dest_folder_list)):
            if not os.path.exists(src_folder) or not os.path.isdir(src_folder):
                print('%s does not exist or it is not a directory, skip ...'%(src_folder))
            if dest_folder!=None and not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            for file in os.listdir(src_folder):
                if file.split('.')[-1] in ['summary',]:
                    src_file=src_folder+os.sep+file
                    dest_file=None if dest_folder==None else dest_folder+os.sep+'.'.join(file.split('.')[:-1])+'.info'
                    file_list.append((src_file,dest_file))
                    sys.stdout.write('%d files are detected\r'%len(file_list))

        print('there are %d files detected in total'%len(file_list))

        for idx,(src_file,dest_file) in enumerate(file_list):
            sys.stdout.write('Process documents %d/%d=%.1f%%\r'%(idx+1,len(file_list),float(idx+1)/float(len(file_list))*100))
            document_info=loader.parse_document(src_file,replace_entity=False) if self.enable_entity_detection==True \
                else loader.parse_document(src_file,replace_entity=True)
            if document_info==None:
                continue

            if format in ['abstractive']:
                idx_document=['1']                                          # start of the documents
                for sentence in document_info['sentences']:
                    idx_sentence=[3,]                                       # start of the sentences
                    sentence=sentence.split(' ')
                    for idx_word, word in enumerate(sentence):
                        if word in document_info['entity2name'] and self.enable_entity_detection:
                            entity_name=document_info['entity2name'][word]
                            if entity_name in self.entity2idx:
                                idx_in_entity_list=self.entity2idx[entity_name]
                                self.entity_frequency[idx_in_entity_list]+=1
                                idx_sentence.append(-(idx_in_entity_list+1))
                            else:
                                assert dictionary_fixed==False, 'Dictionary is fixed while a new word is added'
                                idx_in_entity_list=len(self.entity_list)
                                self.entity_list.append(entity_name)
                                self.entity_frequency.append(1)
                                self.entity2idx[entity_name]=idx_in_entity_list
                                idx_sentence.append(-(idx_in_entity_list+1))
                        else:
                            if word in ['.'] and idx_word==len(sentence)-1:
                                continue
                            if word in self.word2idx:
                                idx_in_word_list=self.word2idx[word]
                                self.word_frequency[idx_in_word_list]+=1
                                idx_sentence.append(idx_in_word_list+1)
                            else:
                                assert dictionary_fixed==False, 'Dictionary is fixed while a new word is added'
                                idx_in_word_list=len(self.word_list)
                                self.word_list.append(word)
                                self.word_frequency.append(1)
                                self.word2idx[word]=idx_in_word_list
                                idx_sentence.append(idx_in_word_list+1)
                    idx_sentence.append(4)                                  # end of the sentences
                    idx_sentence=map(str,idx_sentence)
                    idx_document.append(','.join(idx_sentence))
                idx_document.append('2')                                    # end of the documents

                idx_summary=['1']
                for sentence in document_info['highlights']:
                    idx_sentence=[3,]
                    sentence=sentence.split(' ')
                    for idx_word, word in enumerate(sentence):
                        if word in document_info['entity2name'] and self.enable_entity_detection:
                            entity_name=document_info['entity2name'][word]
                            if entity_name in self.entity2idx:
                                idx_in_entity_list=self.entity2idx[entity_name]
                                self.entity_frequency[idx_in_entity_list]+=1
                                idx_sentence.append(-(idx_in_entity_list+1))
                            else:
                                assert dictionary_fixed==False, 'Dictionary is fixed while a new word is added'
                                idx_in_entity_list=len(self.entity_list)
                                self.entity_list.append(entity_name)
                                self.entity_frequency.append(1)
                                self.entity2idx[entity_name]=idx_in_entity_list
                                idx_sentence.append(-(idx_in_entity_list+1))
                        else:
                            if word in ['.'] and idx_word==len(sentence)-1:
                                continue
                            if word in self.word2idx:
                                idx_in_word_list=self.word2idx[word]
                                self.word_frequency[idx_in_word_list]+=1
                                idx_sentence.append(idx_in_word_list+1)
                            else:
                                assert dictionary_fixed==False, 'Dictionary is fixed while a new word is added'
                                idx_in_word_list=len(self.word_list)
                                self.word_list.append(word)
                                self.word_frequency.append(1)
                                self.word2idx[word]=idx_in_word_list
                                idx_sentence.append(idx_in_word_list+1)
                    idx_sentence.append(4)                                  # end of the sentences
                    idx_sentence=map(str,idx_sentence)
                    idx_summary.append(','.join(idx_sentence))
                idx_summary.append('2')

                if dest_file!=None:
                    with open(dest_file,'w') as fopen:
                        fopen.write(str(len(idx_document)-2)+'\n')
                        fopen.write(','.join(idx_document)+'\n')
                        fopen.write(','.join(idx_summary)+'\n')
            elif format in ['extractive', 'new_extractive']:
                idx_document=[]
                idx_label=map(str,document_info['labels'])
                for sentence in document_info['sentences']:
                    idx_sentence=[]
                    sentence=sentence.split(' ')
                    for word in sentence:
                        if word in document_info['entity2name'] and self.enable_entity_detection:
                            entity_name=document_info['entity2name'][word]
                            if entity_name in self.entity2idx:
                                idx_in_entity_list=self.entity2idx[entity_name]
                                self.entity_frequency[idx_in_entity_list]+=1
                                idx_sentence.append(-(idx_in_entity_list+1))
                            else:
                                assert dictionary_fixed==False, 'Dictionary is fixed while a new word is added'
                                idx_in_entity_list=len(self.entity_list)
                                self.entity_list.append(entity_name)
                                self.entity_frequency.append(1)
                                self.entity2idx[entity_name]=idx_in_entity_list
                                idx_sentence.append(-(idx_in_entity_list+1))
                        else:
                            if word in self.word2idx:
                                idx_in_word_list=self.word2idx[word]
                                self.word_frequency[idx_in_word_list]+=1
                                idx_sentence.append(idx_in_word_list+1)
                            else:
                                assert dictionary_fixed==False, 'Dictionary is fixed while a new word is added'
                                idx_in_word_list=len(self.word_list)
                                self.word_list.append(word)
                                self.word_frequency.append(1)
                                self.word2idx[word]=idx_in_word_list
                                idx_sentence.append(idx_in_word_list+1)
                    idx_sentence=[3,]+idx_sentence+[4,]
                    idx_sentence=map(str,idx_sentence)
                    idx_document.append(idx_sentence)

                if dest_file!=None:
                    with open(dest_file,'w') as fopen:
                        fopen.write(str(len(idx_document))+'\n')
                        for idx_sentence in idx_document:
                            fopen.write(','.join(idx_sentence)+'\n')
                        fopen.write(','.join(idx_label))

            else:
                raise ValueError('Unrecognized format: %s'%format)

        print('Documents loading are completed!        ')

        if dictionary_fixed==False:
            word_list_information=zip(np.arange(1,len(self.word_list)+1),self.word_list,self.word_frequency)
            self.__save_list__(word_list_information,self.word_list_file,list_saved_format)
            if self.enable_entity_detection:
                entity_list_information=zip(np.arange(1,len(self.entity_list)+1),self.entity_list,self.entity_frequency)
                self.__save_list__(entity_list_information,self.entity_list_file,list_saved_format)


    '''
    >>> build index files
    >>> negative number means entity name, positive number means known words
    '''
    def build_idx_files(self,src_folder_list,dest_folder_list, format='abstractive'):
        assert(len(src_folder_list)==len(dest_folder_list))

        file_list=[]
        for idx,(src_folder,dest_folder) in enumerate(zip(src_folder_list,dest_folder_list)):
            if os.path.isdir(src_folder):
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)

                for file in os.listdir(src_folder):
                    if file.split('.')[-1] in ['summary',]:
                        src_file=src_folder+os.sep+file
                        dest_file=dest_folder+os.sep+'.'.join(file.split('.')[:-1])+'.info'
                        file_list.append((src_file,dest_file))
                        sys.stdout.write('%d files are detected\r'%len(file_list))
            elif os.path.isfile(src_folder):
                src_file=src_folder
                dest_file=dest_folder
                if src_file.split('.')[-1] in ['summary']:
                    file_list.append((src_file,dest_file))
                    sys.stdout.write('%d files are detected\r'%len(file_list))
                else:
                    print('Warning: invalid file %s, extended name not recognized'%src_file)
            else:
                print('Warning: invalid file %s, file not exists'%src_file)


        print('there are %d files detected in total'%len(file_list))

        for idx,(src_file,dest_file) in enumerate(file_list):
            sys.stdout.write('process documents %d/%d=%.1f%%\r'%(idx+1,len(file_list),float(idx+1)/float(len(file_list))*100))
            document_info=loader.parse_document(src_file,replace_entity=False) if self.enable_entity_detection==True \
                else loader.parse_document(src_file,replace_entity=True)
            if document_info==None:
                continue

            if format in ['abstractive']:
                idx_document=['1']
                for sentence in document_info['sentences']:
                    idx_sentence=[3]
                    sentence=sentence.split(' ')
                    for word in sentence:
                        if word in document_info['entity2name'] and self.enable_entity_detection:
                            entity_name=document_info['entity2name'][word]
                            if entity_name in self.entity2idx:
                                idx_sentence.append(-(self.entity2idx[entity_name]+1))
                            else:
                                idx_sentence.append(5)
                        else:
                            if word in self.word2idx:
                                idx_sentence.append(self.word2idx[word]+1)
                            else:
                                idx_sentence.append(5)
                    idx_sentence.append(4)
                    idx_sentence=map(str,idx_sentence)
                    idx_document.append(','.join(idx_sentence))
                idx_document.append('2')

                idx_summary=['1']
                for sentence in document_info['highlights']:
                    idx_sentence=[3]
                    sentence=sentence.split(' ')
                    for word in sentence:
                        if word in document_info['entity2name'] and self.enable_entity_detection:
                            entity_name=document_info['entity2name'][word]
                            if entity_name in self.entity2idx:
                                idx_sentence.append(-(self.entity2idx[entity_name]+1))
                            else:
                                idx_sentence.append(5)
                        else:
                            if word in self.word2idx:
                                idx_sentence.append(self.word2idx[word]+1)
                            else:
                                idx_sentence.append(5)
                    idx_sentence.append(4)
                    idx_sentence=map(str,idx_sentence)
                    idx_summary.append(','.join(idx_sentence))
                idx_summary.append('2')

                with open(dest_file,'w') as fopen:
                    fopen.write(str(len(idx_document)-2)+'\n')
                    fopen.write(','.join(idx_document)+'\n')
                    fopen.write(','.join(idx_summary)+'\n')
            elif format in ['extractive', 'new_extractive']:
                idx_document=[]
                idx_label=map(str,document_info['labels'])
                for sentence in document_info['sentences']:
                    idx_sentence=[]
                    sentence=sentence.split(' ')
                    for word in sentence:
                        if word in document_info['entity2name'] and self.enable_entity_detection:
                            entity_name=document_info['entity2name'][word]
                            if entity_name in self.entity2idx:
                                idx_sentence.append(-(self.entity2idx[entity_name]+1))
                            else:
                                idx_sentence.append(5 if format in ['extractive'] else 1)
                        else:
                            if word in self.word2idx:
                                idx_sentence.append(self.word2idx[word]+1)
                            else:
                                idx_sentence.append(5 if format in ['extractive'] else 1)
                    idx_sentence=map(str,idx_sentence)
                    idx_document.append(idx_sentence)

                with open(dest_file,'w') as fopen:
                    fopen.write(str(len(idx_document))+'\n')
                    for idx_sentence in idx_document:
                        fopen.write(','.join(idx_sentence)+'\n')
                    fopen.write(','.join(idx_label))
            else:
                raise ValueError('Uncognized format: %s'%format)

    '''
    >>> from index file to a raw text file
    >>> idx_file, str, index file
    >>> output: (original document, proposed summary)
    '''
    def get_raw_text(self, idx_file, format='asbtractive'):
        with open(idx_file, 'r') as fopen:
            lines=fopen.readlines()
            lines=map(lambda x: x[:-1] if x[-1]=='\n' else x, lines)

            if format in ['asbtractive']:
                assert len(lines)==3, 'There are %d line in file %s, 3 expected'%(len(lines), idx_file)

                num_of_sentences=int(lines[0])
                idx_document=map(int, lines[1].split(','))
                idx_summary=map(int, lines[2].split(','))

                assert idx_document.count(3)==num_of_sentences, 'Expect %d sentences but detect %d <s>'%(
                    num_of_sentences, idx_document.count(3))
                assert idx_document.count(4)==num_of_sentences, 'Expect %d sentences but detect %d </s>'%(
                    num_of_sentences, idx_document.count(4))

                def idx2token(idx):
                    if idx>0:
                        return self.word_list[idx-1]
                    else:
                        return self.entity_list[-idx-1]

                token_document=map(lambda idx: self.word_list[idx-1] if idx>0 else self.entity_list[-idx-1], idx_document)
                token_summary=map(lambda idx: self.word_list[idx-1] if idx>0 else self.entity_list[-idx-1], idx_summary)

                original_document=' '.join(token_document)
                original_summary=' '.join(token_summary)

                return original_document, original_summary
            elif format in ['extractive','new_extractive']:
                num_of_sentences=int(lines[0])
                idx_document=lines[1:-1]
                idx_label=map(int,lines[-1].split(','))
                assert(len(idx_document)==num_of_sentences)

                text_document=[]
                for idx_sentence in idx_document:
                    text_sentence=[]
                    idx_sentence=map(int,idx_sentence.split(','))
                    for idx_word in idx_sentence:
                        if idx_word>=0:         # word
                            if idx_word-1<len(self.word_list):
                                text_word=self.word_list[idx_word-1]
                                if text_word!=None:
                                    text_sentence.append(text_word)
                                else:
                                    raise ValueError('Invalid word idx %d in file %s: hit None value in word_list'%(idx_word,idx_file))
                            else:
                                raise ValueError('Invalid word idx %d in file %s: index exceeds the length of the list'%(idx_word,idx_file))
                        else:                   # entity
                            idx_word=-idx_word
                            if idx_word-1<len(self.entity_list):
                                text_entity=self.entity_list[idx_word-1]
                                if text_word!=None:
                                    text_sentence.append(text_entity)
                                else:
                                    raise ValueError('Invalid word idx %d in file %s: hit None value in entity_list'%(-idx_word,idx_file))
                            else:
                                raise ValueError('Invalid entity idx %d in file %s: index exceeds the length of the list'%(-idx_word, idx_file))
                    text_document.append(' '.join(text_sentence))

                return text_document,idx_label


