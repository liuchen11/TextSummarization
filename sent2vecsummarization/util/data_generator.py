import os
import sys
import random
import collections
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
    import pickle as cPickle
else:
    import cPickle
sys.path.insert(0,'util')
from py2py3 import *
import data_loader
import numpy as np

'''
>>> dataset manager for abstractive model
'''
class data_generator(object):

    '''
    >>> Constructor
    '''
    def __init__(self, hyper_params):
        self.enable_entity_bit=hyper_params['enable_entity_bit'] if 'enable_entity_bit' in hyper_params else None
        self.word_list_length=hyper_params['word_list_length']
        self.entity_list_length=hyper_params['entity_list_length']

        self.word_frequency_threshold=hyper_params['word_frequency_threshold']
        # For abstractive model
        self.max_encoding_step=hyper_params['max_encoding_step']
        self.max_decoding_step=hyper_params['max_decoding_step']

        self.idx_global2local={}
        self.idx_local2global={}
        self.idx_local2text={}
        self.vocab_size=0

        self.word_frequency=None                # list of (index,word,frequency)
        self.entity_frequency=None              # list of (index,word,frequency)
        self.my_data_loader=None                # corresponding data_loader

        self.file_set={}
        self.file_set_pt={}
        self.file_set_subpt={}

    '''
    >>> Private function
    >>> to parse files
    '''
    def __parse_file__(self, file, format):
        if not format.lower() in ['txt','pkl']:
            raise ValueError('Unrecognized file format to parse: %s'%format)

        frequency_list=[]
        if format.lower() in ['txt',]:
            with open(file,'r') as fopen:
                for line in fopen:
                    parts=line.split(' ')
                    idx=int(parts[0])
                    word=' '.join(parts[1:-1])
                    frequency=int(parts[-1])
                    frequency_list.append((idx,word,frequency))
        elif format.lower() in ['pkl',]:
            if sys.version_info.major==2:
                frequency_list=cPickle.load(open(file,'rb'))
            else:
                frequency_list=cPickle.load(open(file,'rb'),encoding='latin1')

        return frequency_list

    '''
    >>> load entity and word information
    >>> entity_file, str, files with entity_name and its frequencies
    >>> dict_file, str, files with word_name and its frequencies
    >>> format, the format of these files, default is None, which means as files' extension mean
    '''
    def load(self, word_file, entity_file, format=None):
        if os.path.exists(word_file):
            print('Word frequency file = %s'%word_file)
            format=format if format!=None else word_file.split('.')[-1]
            self.word_frequency=self.__parse_file__(word_file, format)      # list of [(idx, word, frequency) ...]

            # sort, the first 6 items are special
            if sys.version_info.major==2:
                self.word_frequency[6:]=sorted(self.word_frequency[6:],lambda x,y: 1 if x[2]<y[2] else -1)
            else:
                self.word_frequency[6:]=sorted(self.word_frequency[6:],key=lambda x:x[2],reverse=True)
            # [0, self.word_list_length]
            for sorted_idx,(original_idx,word,frequency) in enumerate(self.word_frequency):
                if sorted_idx>=self.word_list_length or frequency<self.word_frequency_threshold:
                    self.idx_global2local[original_idx]=4                   # 4 means unknown word
                else:
                    self.idx_global2local[original_idx]=sorted_idx
            self.vocab_size=self.word_list_length
        else:
            raise ValueError('Word dictionary file not found: %s'%word_file)

        if entity_file!=None and os.path.exists(entity_file):
            if self.enable_entity_bit==False:
                print('Entity bit is DISABLED!')
                self.my_data_loader=data_loader.data_loader(hyper_params={'word_list_file':word_file,
                    'entity_list_file':entity_file,'enable_entity_detection':False})
                self.my_data_loader.load_dict(list_load_format=format)
            else:
                self.enable_entity_bit=True
                print('Entity bit is ENABLED, file = %s'%entity_file)
                entity_format=format if format!=None else entity_file.split('.')[-1]
                self.entity_frequency=self.__parse_file__(entity_file, entity_format)
                
                # sort
                if sys.version_info.major==2:
                    self.entity_frequency=sorted(self.entity_frequency,lambda x,y: 1 if x[2]<y[2] else -1)
                else:
                    self.entity_frequency=sorted(self.entity_frequency,key=lambda x:x[2],reverse=True)
                # [self.word_list_length, self.word_list_length+self.entity_list_length)
                for sorted_idx,(original_idx,word,frequency) in enumerate(self.entity_frequency):
                    if sorted_idx>=self.entity_list_length:
                        self.idx_global2local[-original_idx]=4
                    else:
                        self.idx_global2local[-original_idx]=self.word_list_length+sorted_idx
                self.vocab_size=self.word_list_length+self.entity_list_length
                self.my_data_loader=data_loader.data_loader(hyper_params={'word_list_file':word_file,
                    'entity_list_file':entity_file,'enable_entity_detection':True})
                self.my_data_loader.load_dict(list_load_format=format)
        else:
            if self.enable_entity_bit==True:
                raise ValueError('Entity dictionary file not found: %s'%entity_file)
            else:
                print('Entity bit is DISABLED!')
                self.my_data_loader=data_loader.data_loader(hyper_params={'word_list_file':word_file,
                    'entity_list_file':entity_file,'enable_entity_detection':False})
                self.my_data_loader.load_dict(list_load_format=format)

    '''
    >>> get local2global idx mapping
    '''
    def get_idx_local2global(self,):
        if len(self.idx_local2global.keys())!=0:
            print('idx_local2global has already been computed')
            return self.idx_local2global

        for global_idx in self.idx_global2local:
            local_idx=self.idx_global2local[global_idx]
            if not local_idx in self.idx_local2global:
                self.idx_local2global[local_idx]=global_idx
            else:
                self.idx_local2global[local_idx]=5              # global idx 5 represents [UNK]

        return self.idx_local2global

    '''
    >>> get local2text mapping
    '''
    def get_idx_local2text(self,):
        if len(self.idx_local2text.keys())!=0:
            print('idx_local2text has already been computed')
            return self.idx_local2text

        if len(self.idx_local2global.keys())==0:
            self.get_idx_local2global()

        for local_idx in self.idx_local2global:
            global_idx=self.idx_local2global[local_idx]
            if global_idx>0:
                word=self.my_data_loader.word_list[global_idx-1]
            else:
                word=self.my_data_loader.entity_list[-global_idx-1]
            self.idx_local2text[local_idx]=word
        return self.idx_local2text

    '''
    >>> dump the global2local index mapping
    '''
    def dump_idx2idx(self, file2dump, format='pkl'):
        if not format.lower() in ['pkl','txt']:
            raise ValueError('invalid format to save dictionary: %s'%format)

        info2dump={'word_list_length':self.word_list_length, 'word_list':[]}

        for global_idx,word,frequency in self.word_frequency:
            local_idx=self.idx_global2local[global_idx]
            info2dump['word_list'].append((word,frequency,global_idx,local_idx))

        if self.enable_entity_bit==True:
            info2dump['entity_list_length']=self.entity_list_length
            info2dump['entity_list']=[]
            for global_idx,entity,frequency in self.entity_frequency:
                local_idx=self.idx_global2local[-global_idx]
                info2dump['entity_list'].append((entity,frequency,-global_idx,local_idx))

        if file2dump!=None:
            if format.lower() in ['pkl',]:
                cPickle.dump(info2dump,open(file2dump,'wb'))
            if format.lower() in ['txt',]:
                with open(file2dump,'w') as fopen:
                    if self.enable_entity_bit==True:
                        fopen.write('%d %d\n'%(self.word_list_length,self.entity_list_length))
                    else:
                        fopen.write('%d\n'%(self.word_list_length))
                    for word,frequency,global_idx,local_idx in info2dump['word_list']:
                        fopen.write('%s %d %d %d\n'%(word,frequency,global_idx,local_idx))
                    if self.enable_entity_bit==True:
                        for entity,frequency,global_idx,local_idx in info2dump['entity_list']:
                            fopen.write('%s %d %d %d\n'%(entity,frequency,global_idx,local_idx))
            print('information saved in %s in %s format'%(file2dump,format))
        return info2dump

    '''
    >>> get the original text from local idx
    >>> idx_info: local idx information
    '''
    def get_text_from_idx(self, idx_info):
        if isinstance(idx_info, collections.Iterable):
            return [self.get_text_from_idx(instance) for instance in idx_info]
        else:
            if len(self.idx_local2text.keys())==0:
                self.get_idx_local2text()
            return self.idx_local2text[idx_info]

    '''
    >>> generator the index file from raw text file
    >>> raw_text_list, list<str>, raw text files
    >>> idx_text_list, list<str>, output index files
    '''
    def gen_idx_from_text(self, raw_text_list, idx_text_list):
        # self.my_data_loader.build_idx_files(src_folder_list=raw_text_list,dest_folder_list=idx_text_list)
        for raw_text, idx_text in zip(raw_text_list,idx_text_list):
            lines=open(raw_text,'r').readlines()
            lines=map(lambda x:x if x[-1]!='\n' else x[:-1],lines)
            idx_document=['1']
            for line in lines:
                idx_sentence=[3]
                for word in line.split(' '):
                    if word in self.my_data_loader.entity2idx:
                        idx_sentence.append(-(self.my_data_loader.entity2idx[word]+1))
                    else:
                        if word in self.my_data_loader.word2idx:
                            idx_sentence.append(self.my_data_loader.word2idx[word]+1)
                        else:
                            idx_sentence.append(5)          # 5 represents unknown words (in idx system beginning from 1)
                idx_sentence.append(4)
                idx_sentence=map(str,idx_sentence)
                idx_document.append(','.join(idx_sentence))
            idx_document.append('2')

            with open(idx_text, 'w') as fopen:
                fopen.write(str(len(idx_document)-2)+'\n')
                fopen.write(','.join(idx_document))
                fopen.write(','.join(['5']*self.max_decoding_step))     # No ground truth summary is provided

    '''
    >>> initialization of data batch for a set of files
    >>> set_label: str, set label e.g. 'train','validate','test'
    >>> file_list: list<str> or None, name of files
    >>> permutation: bool, whether or not to permute the documents
    '''
    def init_batch_gen(self, set_label, file_list, permutation, force=False):
        if set_label in self.file_set and file_list!=None:
            if force==False:
                raise Exception('Can not initialize the dataset %s twice'%set_label)
            #else:
            #    print('reset the dataset %s'%set_label)

        if file_list!=None:
            self.file_set[set_label]=[]
            for file_or_folder in file_list:
                if os.path.isdir(file_or_folder):
                    for file in os.listdir(file_or_folder):
                        if file.split('.')[-1] in ['info',]:
                            self.file_set[set_label].append(file_or_folder+os.sep+file)
                elif os.path.isfile(file_or_folder):
                    if file_or_folder.split('.')[-1] in ['info',]:
                        self.file_set[set_label].append(file_or_folder)
            print('%d files loaded in %s set'%(len(self.file_set[set_label]),set_label))

        self.file_set[set_label]=np.array(self.file_set[set_label])
        self.file_set_pt[set_label]=0
        self.file_set_subpt[set_label]=0
        if permutation==True:
            self.file_set[set_label]=np.random.permutation(self.file_set[set_label])


    '''
    >>> get the training data, including input matrix, masks and labels
    >>> set_label: str, set label e.g. 'train','validate'
    >>> batch_size: int, batch size
    >>> label policy: str in ['min','max','clear'], control if ambiguous sentences are not extracted/extracted/dropped
    >>> model_tag: str, indicating the network type
    >>> output data necessary to train a network
    '''
    def batch_gen(self, set_label, batch_size, label_policy, extend_tags=[], model_tag='abstractive'):
        if not label_policy in ['min','max','clear']:
            raise ValueError('Unrecognized labeling policy %s'%label_policy)
        if not set_label in self.file_set:
            raise ValueError('Set %s has not been initialized yet'%set_label)

        end_of_epoch=False
        if model_tag.lower() in ['abstractive']:
            encode_input_batch=np.zeros([batch_size, self.max_encoding_step], dtype=np.int)
            encode_input_batch.fill(5)                  # 5 represents PAD (idx system beginning with 0)
            encode_input_length=np.zeros([batch_size,], dtype=np.int)
            decode_input_batch=np.zeros([batch_size, self.max_decoding_step], dtype=np.int)
            decode_input_batch.fill(5)                  # 5 represents PAD (idx system beginning with 0)
            decode_refer_batch=np.zeros([batch_size, self.max_decoding_step], dtype=np.int)
            decode_refer_batch.fill(5)                  # 5 represents PAD (idx system beginning with 0)
            decode_mask=np.zeros([batch_size, self.max_decoding_step], dtype=np.int)
            file_list=[]

            for batch_idx in xrange(batch_size):
                dest_file=self.file_set[set_label][self.file_set_pt[set_label]]
                file_list.append(dest_file)
                lines=open(dest_file, 'r').readlines()
                lines=map(lambda x: x[:-1] if x[-1]=='\n' else x, lines)
                assert len(lines)==3, '#lines of the file %s = %d'%(dest_file, len(lines))

                num_of_sentences=int(lines[0])
                idx_this_document=map(int, lines[1].split(','))
                idx_this_summary=map(int, lines[2].split(','))
                assert idx_this_document.count(3)==num_of_sentences, 'There are %d <s> but %d sentences'%(
                    idx_this_document.count(3), num_of_sentences)
                assert idx_this_document.count(4)==num_of_sentences, 'There are %d </s> but %d sentences'%(
                    idx_this_document.count(4), num_of_sentences)

                encode_token_length=min(self.max_encoding_step, len(idx_this_document))
                decode_token_legnth=min(self.max_decoding_step, len(idx_this_summary))
                decode_refer_length=min(self.max_decoding_step, len(idx_this_summary[1:]))

                idx_encode_input=map(lambda x: self.idx_global2local[x], idx_this_document[:encode_token_length])
                idx_decode_input=map(lambda x: self.idx_global2local[x], idx_this_summary[:decode_token_legnth])
                idx_decode_refer=map(lambda x: self.idx_global2local[x], idx_this_summary[1:decode_refer_length+1])

                encode_input_batch[batch_idx,:encode_token_length]=idx_encode_input
                encode_input_length[batch_idx]=encode_token_length
                decode_input_batch[batch_idx,:decode_token_legnth]=idx_decode_input
                decode_refer_batch[batch_idx,:decode_refer_length]=idx_decode_refer
                decode_mask[batch_idx,:decode_refer_length].fill(1)

                self.file_set_pt[set_label]+=1
                if self.file_set_pt[set_label]>=len(self.file_set[set_label]):
                    end_of_epoch=True
                    self.init_batch_gen(set_label=set_label, file_list=None, permutation=True)

            return {'encode_input_batch':encode_input_batch, 'encode_input_length':encode_input_length, 'decode_input_batch':decode_input_batch,
                'decode_refer_batch':decode_refer_batch, 'decode_mask':decode_mask, 'end_of_epoch':end_of_epoch, 'file_list':file_list}
        elif model_tag.lower() in ['extractive', 'new_extractive']:
            input_matrix=np.zeros([batch_size, self.max_encoding_step], dtype=np.int)
            if model_tag.lower() in ['extractive']:
                input_matrix.fill(5)
            if model_tag.lower() in ['new_extractive']:
                input_matrix.fill(1)
            masks=np.zeros([batch_size, self.max_encoding_step], dtype=np.int)
            labels=np.zeros([batch_size,], dtype=np.int)

            extension_part={}
            for extend_tag in extend_tags:
                if extend_tag.lower() in ['entity_bit']:
                    extension_part['entity_bit']=np.zeros([batch_size, self.max_encoding_step, 1], dtype=np.float32)
                else:
                    raise ValueError('Unrecognized extend tag: %s'%extend_tag)

            dest_file=self.file_set[set_label][self.file_set_pt[set_label]]
            lines=open(dest_file,'r').readlines()
            lines=map(lambda x: x[:-1] if x[-1]=='\n' else x,lines)
            number_of_sentences=int(lines[0])
            idx_this_document=map(lambda x:map(int,x.split(',')),lines[1:-1])
            labels_this_document=map(int,lines[-1].split(','))
            assert(number_of_sentences+2==len(lines))

            batch_idx=0
            while batch_idx<batch_size:
                label_this_sentence=labels_this_document[self.file_set_subpt[set_label]]
                if label_this_sentence!=2 or not label_policy in ['clear',]:
                    idx_sentence=idx_this_document[self.file_set_subpt[set_label]]
                    if len(idx_sentence)>self.max_encoding_step:
                        off_set=random.randint(0,len(idx_sentence)-self.max_encoding_step)
                        idx_sentence=idx_sentence[off_set:off_set+self.max_encoding_step]
                    local_idx_sentence=map(lambda x:self.idx_global2local[x],idx_sentence)
                    input_matrix[batch_idx,:len(idx_sentence)]=local_idx_sentence

                    if 'entity_bit' in extension_part:
                        entity_bit_sentence=map(lambda x: 1 if x<0 else 0, idx_sentence)
                        extension_part['entity_bit'][batch_idx,:len(idx_sentence),:]=np.array(entity_bit_sentence).reshape([len(idx_sentence),1])

                    masks[batch_idx,:len(local_idx_sentence)].fill(1)
                    if label_this_sentence==2:
                        label_this_sentence=1 if label_policy in ['max',] else 0
                    labels[batch_idx]=label_this_sentence
                    batch_idx+=1

                # Next sentence
                self.file_set_subpt[set_label]+=1
                if self.file_set_subpt[set_label]==number_of_sentences:
                    self.file_set_pt[set_label]+=1
                    self.file_set_subpt[set_label]=0
                    if self.file_set_pt[set_label]==len(self.file_set[set_label]):
                        end_of_epoch=True
                        self.init_batch_gen(set_label=set_label, file_list=None, permutation=True)
                    dest_file=self.file_set[set_label][self.file_set_pt[set_label]]
                    lines=open(dest_file,'r').readlines()
                    lines=map(lambda x: x[:-1] if x[-1]=='\n' else x,lines)
                    number_of_sentences=int(lines[0])
                    idx_this_document=map(lambda x:map(int,x.split(',')),lines[1:-1])
                    labels_this_document=map(int,lines[-1].split(','))
                    assert(number_of_sentences+2==len(lines))

            return {'input_matrix':input_matrix, 'masks':masks, 'labels':labels, 'end_of_epoch':end_of_epoch, 'extension_part':extension_part}
        else:
            raise ValueError('Unrecognized model tag: %s'%model_tag)
