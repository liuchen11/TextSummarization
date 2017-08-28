import os
import sys
if sys.version_info.major==3:
    xrange=range
import numpy as np
import tensorflow as tf

class BeamSearchHandler(object):

    def __init__(self, hyper_params):
        '''
        >>> hyper_params
            >>> beam_size: int, the number of candidate sequence
            >>> model: PGNet model
            >>> generator: data generator, the source of data
            >>> single_pass: bool, whether go through only one epoch or more
        '''
        self.beam_size=hyper_params['beam_size']
        self.model=None                 # need to be bind
        self.generator=None             # need to be bind
        self.single_pass=True           # self.single_pass is set True by default

    def bind_model(self, model):
        '''
        >>> bind a model to this BeamSearchHandler
        >>> model: PGNet model
        '''
        assert model.mode=='decode', 'The mode of the model has to be "decode", while now it is %s'%model.mode
        assert model.max_decoding_step==1, 'The max decoding step has to be 1, while now it is %d'%model.max_decoding_step
        assert model.batch_size==self.beam_size, 'The network\'s batch_size must be consistent with the beam_size,\
         but now they are %d and %d respectively'%(model.batch_size, self.beam_size)
        if self.generator!=None:
            assert self.generator.max_encoding_step==model.max_encoding_step, \
            'The max_encoding_step of the generator and the model must be the same, but now they are %d and %d respectively'%(self.generator.max_encoding_step, model.max_encoding_step)
            assert self.generator.vocab_size==model.vocab_size, \
            'The vocab_size of the generator and the model must be the same, but now they are %d and %d respectively'%(self.generator.max_encoding_step, model.max_encoding_step)
        if self.model!=None:
            print('Current model %s will be replaced'%self.model.name)
        self.model=model
        print('Model %s has been loaded'%self.model.name)

    def bind_generator(self, generator):
        '''
        >>> bind a generator to this BeamSearchHandler
        >>> generator: data generator
        '''
        if self.model!=None:
            assert self.model.max_encoding_step==generator.max_encoding_step, \
            'The max_encoding_step of the generator and the model must be the same, but now they are %d and %d respectively'%(\
                generator.max_encoding_step, self.model.max_encoding_step)
            assert self.model.vocab_size==generator.vocab_size, \
            'The vocab_size of the generator and the model must be the same, but now they are %d and %d respectively'%(\
                generator.max_encoding_step, self.model.max_encoding_step)
        if self.generator!=None:
            print('Current data generator %s will be replaced')
        self.generator=generator
        print('Data generator has been loaded')

    def _clean_text(self, text):
        '''
        >>> clear some tags like [START] [END]
        '''
        text=text.replace('[START]','')
        text=text.replace('[END]','')
        text=text.replace('[PAD]','')
        text=text.replace('<s>','')
        text=text.replace('</s>','\n')
        return text

    def decode(self, set_label, file_list, fout, single_pass=True, max_processed_documents=-1):
        '''
        >>> decoding function

        >>> set_label: str, name of the document set to be decoded
        >>> file_list: list<str>, file or folder list
        >>> fout: output file
        >>> single_pass: bool, to decode the document at most once
        >>> max_processed_documents: int, maximum number of documents to be decoded, negative number means infinity
        '''
        print('Start generating the sentences!')

        if single_pass==False and max_processed_documents<0:
            raise ValueError('When the single is disabled, max_processed_documents can not set to be negative')

        # Print the output information
        if fout in [None, sys.stdout]:
            print('Outputs will be in the command line')
        else:
            if os.path.isfile(fout):
                print('Outputs will be a single file: %s'%fout)
            else:
                if not os.path.exists(fout):
                    os.makedirs(fout)
                else:
                    print('Outputs will under the folder: %s'%fout)

        self.generator.init_batch_gen(set_label=set_label, file_list=file_list, permutation=True)
        num_processed_documents=0

        while True:
            sys.stdout.write('Decoding the documents in set %s: %d\r'%(set_label, num_processed_documents+1))
            document_info=self.generator.batch_gen(set_label=set_label, batch_size=1, label_policy='min', extend_tags=[], model_tag='abstractive')

            decode_beam=[[0] for _ in xrange(self.beam_size)]                                                       # candidate sequence in decode_beam
            decode_log_likely=[0 for _ in xrange(self.beam_size)]                                                   # likelihood of each candidate sequence
            decode_hidden_states=np.array([],dtype=np.float32).reshape([self.model.batch_size,0,self.model.decoding_dim])            # decode hidden states, initialized as empty
            encode_att_history=np.ones([self.model.batch_size, self.model.max_encoding_step], dtype=np.float32)    # intra-encoding attention normalizer

            # run encoder to get the initial decoding states
            encode_input_batch_v=np.repeat(document_info['encode_input_batch'], repeats=self.model.batch_size, axis=0)    # batch_size dimension broadcast
            encode_input_length_v=np.repeat(document_info['encode_input_length'], repeats=self.model.batch_size, axis=0)  # batch_size dimension broadcast

            encoding_states_v, init_decode_state_v=self.model.run_encoder(encode_input_batch_v=encode_input_batch_v, encode_input_length_v=encode_input_length_v)
            for decode_idx in xrange(self.generator.max_decoding_step):
                # print('==== %d ===='%decode_idx)
                # get the input tokens
                input_token=np.array([seq[-1:] for seq in decode_beam])
                decode_state_v, h_decode_states_v, encode_att_final_v, top_value_v, top_idx_v=self.model.decode_one_step(
                    encode_input_batch_v=encode_input_batch_v, init_decode_state_v=init_decode_state_v, lastest_token=input_token,
                    encoding_states_v=encoding_states_v, decoding_states_v=decode_hidden_states, encode_att_init_v=encode_att_history)
                init_decode_state_v=decode_state_v

                # Pick up the top beam_size candidate
                beam_list=[]
                for batch_idx in xrange(self.model.batch_size):
                    if decode_idx==0 and batch_idx>0:           # In the first time stamp, the input candidates are exactly the same, so no need to analyze more than 1 output instances
                        break

                    if decode_beam[batch_idx].count(1)!=0:      # already finished
                        if len(decode_beam[batch_idx])>=8:      # Too short
                            beam_list.append([batch_idx,decode_beam[batch_idx],decode_log_likely[batch_idx],])
                    else:
                        for word_candidate_idx in xrange(self.model.batch_size):
                            word_idx=top_idx_v[batch_idx,0,word_candidate_idx]
                            word_prob=top_value_v[batch_idx,0,word_candidate_idx]
                            log_likely_this_instance=(decode_log_likely[batch_idx]*decode_idx+word_prob)/(decode_idx+1)
                            sequence_this_instance=decode_beam[batch_idx]+[word_idx,]
                            beam_list.append([batch_idx,sequence_this_instance,log_likely_this_instance,])
                if sys.version_info.major==2:
                    beam_list=sorted(beam_list, lambda x,y: 1 if x[2]<y[2] else -1)
                else:
                    beam_list=sorted(beam_list, key=lambda x:x[2], reverse=True)

                beam_list=beam_list[:self.beam_size]
                decode_beam=[instance[1] for instance in beam_list]
                decode_log_likely=[instance[2] for instance in beam_list]

                new_decode_hidden_states=np.zeros([self.model.batch_size, decode_idx+1, self.model.decoding_dim],dtype=np.float32)
                new_encode_att_history=np.ones([self.model.batch_size, self.model.max_encoding_step], dtype=np.float32)
                for batch_idx in xrange(self.model.batch_size):
                    ori_batch_idx=beam_list[batch_idx][0]
                    if decode_idx>0:
                        new_decode_hidden_states[batch_idx,:decode_idx]=decode_hidden_states[ori_batch_idx]
                    new_decode_hidden_states[batch_idx,decode_idx:]=h_decode_states_v[ori_batch_idx].reshape([1,self.model.decoding_dim])
                    new_encode_att_history[batch_idx]=encode_att_final_v[ori_batch_idx]
                decode_hidden_states=new_decode_hidden_states
                encode_att_history=new_encode_att_history

            # Generate summary
            ori_document_idx=document_info['encode_input_batch'][0]
            ori_document_text=' '.join(self.generator.get_text_from_idx(ori_document_idx))
            ori_document_text=self._clean_text(ori_document_text)
            gold_summary_idx=document_info['decode_refer_batch'][0]
            gold_summary_text=' '.join(self.generator.get_text_from_idx(gold_summary_idx))
            gold_summary_text=self._clean_text(gold_summary_text)
            proposed_summary_idx=beam_list[0][1]
            proposed_summary_text=' '.join(self.generator.get_text_from_idx(proposed_summary_idx))
            proposed_summary_text=self._clean_text(proposed_summary_text)
            num_processed_documents+=1

            if fout in [None, sys.stdout]:
                print('=========Original Document=========')
                print(ori_document_text)
                print('===========Gold Summary============')
                print(gold_summary_text)
                print('==========Proposed Summary=========')
                print(proposed_summary_text)
            else:
                if os.path.isfile(fout):        # dump all the information into a single file
                    with open(fout,'a') as fopen:
                        fopen.write('=========Original Document=========\n')
                        fopen.write('%s\n'%ori_document_text)
                        fopen.write('===========Gold Summary============\n')
                        fopen.write('%s\n'%gold_summary_text)
                        fopen.write('==========Proposed Summary=========\n')
                        fopen.write('%s\n'%proposed_summary_text)
                else:                           # save all the information under a folder
                    if not os.path.exists(fout):
                        os.makedirs(fout)
                    with open(fout+os.sep+'%d.txt'%num_processed_documents,'w') as fopen:
                        fopen.write('=========Original Document=========\n')
                        fopen.write('%s\n'%ori_document_text)
                        fopen.write('===========Gold Summary============\n')
                        fopen.write('%s\n'%gold_summary_text)
                        fopen.write('==========Proposed Summary=========\n')
                        fopen.write('%s\n'%proposed_summary_text)

            if (single_pass and document_info['end_of_epoch']) or (max_processed_documents>0 and max_processed_documents<=num_processed_documents):
                print('Decoding process has finished!')
                break











