import os
import sys
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
    import pickle as cPickle
else:
    import cPickle
sys.path.insert(0,'util')
from py2py3 import *
import loader
import data_loader
import random
import numpy as np

'''
>>> dataset manager
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
        self.document_length_threshold=hyper_params['document_length_threshold']
        self.sentence_length_threshold=hyper_params['sentence_length_threshold']
        self.idx_global2local={}
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
            self.word_frequency=self.__parse_file__(word_file, format)

            # sort
            if sys.version_info.major==2:
                self.word_frequency=sorted(self.word_frequency,lambda x,y: 1 if x[2]<y[2] else -1)
            else:
                self.word_frequency=sorted(self.word_frequency,key=lambda x:x[2],reverse=True)
            # [0,self.word_list_length+1) is reserved for ordinary word
            for sorted_idx,(original_idx,word,frequency) in enumerate(self.word_frequency):
                if sorted_idx>=self.word_list_length or frequency<self.word_frequency_threshold:
                    self.idx_global2local[original_idx]=self.word_list_length
                else:
                    self.idx_global2local[original_idx]=sorted_idx
            self.idx_global2local[int(1e10)]=self.word_list_length          # unknown words are marked 1e10
            self.vocab_size=self.word_list_length+1
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
                # [self.word_list_length+1, self.word_list_length+self.entity_list_length+2)
                for sorted_idx,(original_idx,word,frequency) in enumerate(self.entity_frequency):
                    if sorted_idx>=self.entity_list_length:
                        self.idx_global2local[-original_idx]=self.word_list_length+1+self.entity_list_length
                    else:
                        self.idx_global2local[-original_idx]=self.word_list_length+1+sorted_idx
                self.idx_global2local[-int(1e10)]=self.word_list_length+1+self.entity_list_length    # unknown entity are marked -1e10
                self.vocab_size=self.word_list_length+self.entity_list_length+2
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
    >>> dump the global2local index mapping
    '''
    def dump_idx2idx(self,file2dump,format='pkl'):
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
            print('information saved in %s in %d format'%(file2dump,format))
        return info2dump

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
            idx_document=[]
            for line in lines:
                idx_sentence=[]
                for word in line.split(' '):
                    if word in self.my_data_loader.entity2idx:
                        idx_sentence.append(-(self.my_data_loader.entity2idx[word]+1))
                    else:
                        if word in self.my_data_loader.word2idx:
                            idx_sentence.append(self.my_data_loader.word2idx[word]+1)
                        else:
                            idx_sentence.append(int(1e10))
                idx_sentence=map(str,idx_sentence)
                idx_document.append(idx_sentence)

            fake_label=['0' for _ in idx_document]

            with open(idx_text, 'w') as fopen:
                fopen.write(str(len(idx_document))+'\n')
                for idx_sentence in idx_document:
                    fopen.write(','.join(idx_sentence)+'\n')
                fopen.write(','.join(fake_label))

    '''
    >>> initialization of data batch for a set of files
    >>> set_label: str, set label e.g. 'train','validate','test'
    >>> file_list: list<str> or None, name of files
    >>> permutation: bool, whether or not to permute the documents
    '''
    def init_batch_gen(self,set_label,file_list,permutation, force=False):
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
            #print('%d files loaded in %s set'%(len(self.file_set[set_label]),set_label))

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
    def batch_gen(self,set_label,batch_size,label_policy,extend_tags=[],model_tag='sentence_extract'):
        if not label_policy in ['min','max','clear']:
            raise ValueError('Unrecognized labeling policy %s'%label_policy)
        if not set_label in self.file_set:
            raise ValueError('Set %s has not been initialized yet'%set_label)

        end_of_epoch=False
        if model_tag.lower() in ['sentence_extract',]:
            input_matrix=np.zeros([batch_size,self.document_length_threshold,self.sentence_length_threshold],dtype=np.int)
            input_matrix.fill(self.vocab_size)
            masks=np.zeros([batch_size,self.document_length_threshold],dtype=np.int)
            labels=np.zeros([batch_size,self.document_length_threshold],dtype=np.int)

            extension_part={}
            for extend_tag in extend_tags:
                if extend_tag.lower() in ['entity_bit']:
                    extension_part['entity_bit']=np.zeros([batch_size,self.document_length_threshold,self.sentence_length_threshold,1],dtype=np.float32)
                else:
                    raise ValueError('Unrecognized extend tag: %s'%extend_tag)

            for batch_idx in xrange(batch_size):
                dest_file=self.file_set[set_label][self.file_set_pt[set_label]]
                lines=open(dest_file,'r').readlines()
                lines=map(lambda x: x[:-1] if x[-1]=='\n' else x, lines)
                number_of_sentences=int(lines[0])
                idx_this_document=map(lambda x:map(int,x.split(',')),lines[1:-1])
                labels_this_document=map(int,lines[-1].split(','))
                assert(number_of_sentences+2==len(lines))

                if number_of_sentences>self.document_length_threshold:
                    offset=random.randint(0,number_of_sentences-self.document_length_threshold)
                    labels_this_document=labels_this_document[offset:offset+self.document_length_threshold]
                    idx_this_document=idx_this_document[offset:offset+self.document_length_threshold]
                    number_of_sentences=self.document_length_threshold

                masks_this_document=np.ones([number_of_sentences],dtype=np.int)
                for idx,idx_sentence in enumerate(idx_this_document):
                    if len(idx_sentence)>self.sentence_length_threshold:
                        offset=random.randint(0,len(idx_sentence)-self.sentence_length_threshold)
                        idx_sentence=idx_sentence[offset:offset+self.sentence_length_threshold]
                    local_idx_sentence=map(lambda x:self.idx_global2local[x],idx_sentence)
                    input_matrix[batch_idx,idx,:len(idx_sentence)]=local_idx_sentence

                    if 'entity_bit' in extension_part:
                        entity_bit_sentence=map(lambda x: 1 if x<0 else 0,idx_sentence)
                        extension_part['entity_bit'][batch_idx,idx,:len(idx_sentence),:]=np.array(entity_bit_sentence).reshape([len(idx_sentence),1])

                for idx,(label,mask) in enumerate(zip(labels_this_document,masks_this_document)):
                    if label==2:
                        if label_policy in ['min',]:
                            labels_this_document[idx]=0
                        elif label_policy in ['max',]:
                            labels_this_document[idx]=1
                        elif label_policy in ['clear',]:
                            masks_this_document[idx]=0

                masks[batch_idx,:number_of_sentences]=masks_this_document
                labels[batch_idx,:number_of_sentences]=labels_this_document

                self.file_set_pt[set_label]+=1
                if self.file_set_pt[set_label]==len(self.file_set[set_label]):
                    end_of_epoch=True
                    self.init_batch_gen(set_label,file_list=None,permutation=True)

            return input_matrix,masks,labels,end_of_epoch,extension_part
        elif model_tag.lower() in ['fasttext','fast_text','naive']:
            input_matrix=np.zeros([batch_size, self.sentence_length_threshold],dtype=np.int)
            input_matrix.fill(self.vocab_size)
            masks=np.zeros([batch_size, self.sentence_length_threshold],dtype=np.int)
            labels=np.zeros([batch_size,],dtype=np.int)

            extension_part={}
            for extend_tag in extend_tags:
                if extend_tag.lower() in ['entity_bit']:
                    extension_part['entity_bit']=np.zeros([batch_size,self.sentence_length_threshold,1],dtype=np.float32)
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
                    if len(idx_sentence)>self.sentence_length_threshold:
                        off_set=random.randint(0,len(idx_sentence)-self.sentence_length_threshold)
                        idx_sentence=idx_sentence[off_set:off_set+self.sentence_length_threshold]
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
                        self.init_batch_gen(set_label=set_label,file_list=None,permutation=True)
                    dest_file=self.file_set[set_label][self.file_set_pt[set_label]]
                    lines=open(dest_file,'r').readlines()
                    lines=map(lambda x: x[:-1] if x[-1]=='\n' else x,lines)
                    number_of_sentences=int(lines[0])
                    idx_this_document=map(lambda x:map(int,x.split(',')),lines[1:-1])
                    labels_this_document=map(int,lines[-1].split(','))
                    assert(number_of_sentences+2==len(lines))

            return input_matrix,masks,labels,end_of_epoch,extension_part
        else:
            raise ValueError('Unrecognized model tag: %s'%model_tag)
