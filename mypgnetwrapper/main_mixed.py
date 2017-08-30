import os
import sys
if sys.version_info.major==2:
    input=raw_input
else:
    from builtins import input
import socket
import select

import tensorflow as tf
import numpy as np
import traceback
from collections import namedtuple

file_path=os.path.dirname(os.path.realpath(__file__))
file_path=os.path.join(os.path.join(file_path,os.pardir),'model')
sys.path.insert(0, file_path)
import util
import html
import data
import model
import batcher
import beam_search

def encode_sth(item):
    coding=['iso-8859-1','utf8','latin1','ascii']
    for coding_format in coding:
        try:
            coded=item.encode(coding_format)
            return coded
        except:
            continue
    raise Exception('Unable to encode',item)

def decode_sth(item):
    coding=['iso-8859-1','utf8','latin1','ascii']
    for coding_format in coding:
        try:
            coded=item.decode(coding_format)
            return coded
        except:
            continue
    raise Exception('Unable to decode',item)

FLAGS=tf.app.flags.FLAGS

'''
>>> hyper_param field
'''
# Where to find data
tf.app.flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode. If True, go through the dataset exactly once then stop. Otherwise, loop through the dataset randomly indefinitely')

# Where to save output
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')
tf.app.flags.DEFINE_string('interactive_mode', 'cs', 'Interactive mode')

# Sentence Extract baseline
tf.app.flags.DEFINE_string('sentence_extract_root','.','root directory of the sentence extraction model')
tf.app.flags.DEFINE_string('sentence_extract_config','.','config file of the sentence extraction model')

class RunTimeWrapper(object):

    '''
    >>> initialization and construction of the model
    '''
    def __init__(self, hp, model_settings, extra_info):
        vocab_file=hp.vocab_path
        max_size=hp.vocab_size
        self.vocab=data.Vocab(vocab_file=vocab_file, max_size=max_size)     # Construct the vocabulary manager

        # model_hp_list=['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
        #     'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt', 'pointer_gen']
        # model_hp_dict={}
        # for key,value in FLAGS.__flags.iteritems():
        #     if key in model_hp_list:
        #         model_hp_dict[key]=value
        # model_settings=namedtuple('HParams',model_hp_dict.keys())(**model_hp_dict)

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
        print(best_list.tokens)
        output_ids=[int(t) for t in best_list.tokens[1:]]
        decoded_words=data.outputids2words(output_ids,self.vocab,(batch.art_oovs[0] if self.hp.pointer_gen==True else None))
        print(decoded_words)

        try:
            fst_stop_idx=decoded_words.index(data.STOP_DECODING)
            decoded_words=decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words=decoded_words
        decoded_output=' '.join(decoded_words)

        return article_withunks,abstract_withunks,decoded_output

def main(unused_argv):
    if len(unused_argv)!=1:
        raise Exception('Problem with flags: %s'%str(unused_argv))

    # start sentence extraction model
    ret_path=os.path.abspath('.')
    os.chdir(FLAGS.sentence_extract_root)
    sys.path.insert(0,'./run')
    sys.path.insert(0,'./util')
    import laucher
    import xml_parser

    laucher_params=xml_parser.parse(FLAGS.sentence_extract_config,flat=False)
    se_solver=laucher.laucher(laucher_params)
    se_solver.start()
    os.chdir(ret_path)

    loaded_params=tf.global_variables()
    try:
        assert(FLAGS.mode=='decode')
    except:
        raise ValueError('mode must be "decode" but now it is %s'%str(FLAGS.mode))
    FLAGS.log_root=os.path.join(FLAGS.log_root,FLAGS.exp_name)
    try:
        assert(os.path.exists(FLAGS.log_root))
    except:
        raise ValueError('Invalid log_root: %s'%str(FLAGS.log_root))
    FLAGS.batch_size=FLAGS.beam_size

    model_hp_list=['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
        'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt', 'pointer_gen']
    model_hp_dict={}
    for key,value in FLAGS.__flags.iteritems():
        if key in model_hp_list:
            model_hp_dict[key]=value
    model_settings=namedtuple('HParams',model_hp_dict.keys())(**model_hp_dict)
    model_settings=model_settings._replace(max_dec_steps=1)

    solver=RunTimeWrapper(hp=FLAGS, model_settings=model_settings)
    solver.start(loaded_params=loaded_params)
    if FLAGS.interactive_mode=='cmd':
        command_line_mode(solver,se_solver)
    elif FLAGS.interactive_mode=='cs':
        server_client_mode(solver,se_solver)
    else:
        raise ValueError('Unrecognized interative mode: %s'%FLAGS.interactive_mode)

    solver.end()
    se_solver.end()

def command_line_mode(solver,se_solver):
    while True:
        print('Give a URL and get a summary, type "exit" to abort')
        answer=input('>>>')
        if answer=='exit':
            break

        try:
            raw_content=html.get_content_from_url(answer)
            raw_sentences=raw_content.split('\n')
            sentence_list=[]
            for sentence in raw_sentences:
                if len(sentence.split(' '))>=5:
                    sentence_list.append(sentence)
            text_content='\n'.join(sentence_list)
            with open('tmp.txt','w') as fopen:
                fopen.write(text_content)
            selected_sentence=se_solver.select('tmp.txt')
            content=selected_sentence.replace('\n',' ').lower()
            article,summary=solver.run(query=content)
            print(article)
        except:
            print('Oops, some problems occurs while loading contents from %s'%answer)
            traceback.print_exc()
            continue
        print('===============Article=============')
        print(content)
        print('===============Summary=============')
        print(summary)
        print('=================End===============')

def server_client_mode(solver,se_solver):
    # Open a socket
    host='127.0.0.1'
    port=6100

    socket_list=[]
    try:
        server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        server.bind((host,port))
        server.listen(5)
        socket_list.append(server)
        print('Server/Client mode start! Listening on %s:%s'%(host,port))
    except:
        traceback.print_exc()
        print('Unable to start the server. Abort!')
        exit(1)

    while True:
        ready2read,ready2write,in_err=select.select(socket_list,[],[],0)

        for sock in ready2read:
            if sock==server:        # New connection
                sockfd,addr=server.accept()
                socket_list.append(sockfd)
                print('Client (%s,%s) connected'%addr)
            else:                   # Message from a client
                try:
                    data=encode_sth(sock.recv(1024))
                    if data:
                        try:
                            data=data.rstrip('\n')
                            print('Analyzing content on %s'%data)
                            raw_content=html.get_content_from_url(data)
                            sentences=raw_content.split('\n')
                            sentence_list=[]
                            sentence_list_with_idx=[]
                            for sentence in sentences:
                                if len(sentence.split(' '))>=5:
                                    sentence_list.append(sentence)
                                    sentence_list_with_idx.append('[%d] %s'%(len(sentence_list),sentence))
                            content='\n'.join(sentence_list)
                            content_with_idx='\n'.join(sentence_list_with_idx)
                            ret_path=os.path.abspath('.')
                            os.chdir(FLAGS.sentence_extract_root)
                            with open('tmp.txt','w') as fopen:
                                fopen.write(content)
                            selected_sentence=se_solver.select('tmp.txt')
                            os.system('rm tmp.txt')
                            os.chdir(ret_path)
                            content=selected_sentence.replace('\n',' ').lower()
                            article,summary=solver.run(query=content)
                            message='%s@@@@@%s'%(encode_sth(content_with_idx),encode_sth(summary))
                            sock.send(message)
                            print('Completed!')
                        except:
                            traceback.print_exc()
                            message='Oops, some problems occurs while loading contents from %s'%data
                            print(message)
                            sock.send(message+'\n')
                            continue
                    else:
                        if sock in socket_list:
                            socket_list.remove(sock)
                        print('One server is offline')
                except:
                    traceback.print_exc()
                    print('Some error happens')
                    break

    server.close()


if __name__=='__main__':
    tf.app.run()

