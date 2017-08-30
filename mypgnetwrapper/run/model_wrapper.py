import os
import sys

import tensorflow as tf

file_path=os.path.dirname(os.path.realpath(__file__))
model_path=os.path.join(os.path.join(file_path,os.pardir),'model')
sys.path.insert(0, model_path)
import util
import html
import data
import model
import batcher
import beam_search

FLAGS=tf.app.flags.FLAGS

class RunTimeWrapper(object):

    '''
    >>> initialization and construction of the model
    '''
    def __init__(self, hp, model_settings, extra_info, mode='decode'):
        vocab_file=hp.vocab_path
        max_size=hp.vocab_size
        self.vocab=data.Vocab(vocab_file=vocab_file, max_size=max_size)     # Construct the vocabulary manager

        self.model=model.SummarizationModel(hps=model_settings, vocab=self.vocab, extra_info=extra_info)   # Construct the model
        self.decode_wrapper=None

    '''
    >>> launch the tensorflow session
    '''
    def start(self,loaded_params=[]):
        if self.decode_wrapper!=None:
            raise ValueError('Start function is called only if decode wrapper is None')
        self.decode_wrapper=DecoderWrapper(self.model,self.vocab,hp=FLAGS,loaded_params=loaded_params)

    '''
    >>> process a query
    '''
    def run(self, query):
        batch=build_batch(query=query,vocab=self.vocab,hp=FLAGS)
        article,_,summary=self.decode_wrapper.run_beam_decoder(batch)
        return article,summary

    '''
    >>> close the tensorflow session
    '''
    def end(self,):
        if self.decode_wrapper==None:
            raise ValueError('End function is called only if decode wrapper is non-None')
        self.decode_wrapper=None

'''
>>> from the raw text 'article' to generate a batch recognizable by model
'''
def build_batch(query,vocab,hp):
    instance=batcher.Example(article=query, abstract_sentences='', vocab=vocab, hps=hp)
    batch=batcher.Batch(example_list=[instance,]*hp.batch_size, hps=hp, vocab=vocab)
    return batch

class DecoderWrapper(object):

    '''
    >>> wrapper for the decoder
    >>> model: summarization model
    >>> vocab: vocabulary object
    '''
    def __init__(self,model,vocab,hp,loaded_params=[]):
        self.hp=hp
        self.model=model
        self.sess=tf.Session(config=util.get_config())
        self.vocab=vocab

        self.model.build_graph()
        param2load=[]
        for var in tf.global_variables():
            if not var in loaded_params:
                param2load.append(var)
        ckpt_path=util.load_ckpt(tf.train.Saver(param2load),self.sess)        # load the value of saved parameters
        self.decode_dir=os.path.join(self.hp.log_root, 'decode')

        if not os.path.exists(self.decode_dir):
            os.makedirs(self.decode_dir)

    '''
    >>> run beam decoder
    >>> batch: batch data
    '''
    def run_beam_decoder(self,batch):
        original_article=batch.original_articles[0]
        original_abstract=batch.original_abstracts[0]

        article_withunks=data.show_art_oovs(original_article,self.vocab)
        abstract_withunks=data.show_abs_oovs(original_abstract,self.vocab,(batch.art_oovs[0] if self.hp.pointer_gen==True else None))

        best_list=beam_search.run_beam_search(self.sess,self.model,self.vocab,batch)
        output_ids=[int(t) for t in best_list.tokens[1:]]
        decoded_words=data.outputids2words(output_ids,self.vocab,(batch.art_oovs[0] if self.hp.pointer_gen==True else None))

        try:
            fst_stop_idx=decoded_words.index(data.STOP_DECODING)
            decoded_words=decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words=decoded_words
        decoded_output=' '.join(decoded_words)

        return article_withunks,abstract_withunks,decoded_output
