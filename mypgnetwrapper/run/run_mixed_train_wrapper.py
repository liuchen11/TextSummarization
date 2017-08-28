import os
import sys
if sys.version_info.major==2:
    import cPickle
else:
    import pickle as cPickle

import numpy as np
import tensorflow as tf
import traceback
from nltk import tokenize
from collections import namedtuple
from random import shuffle

file_path=os.path.dirname(os.path.realpath(__file__))
model_path=os.path.join(os.path.join(file_path,os.pardir),'model')
util_path=os.path.join(os.path.join(file_path,os.pardir),'util')
sys.path.insert(0, model_path)
sys.path.insert(0, util_path)

import util
import data
import model
import batcher
import beam_search

from data_wrapper import *
from model_wrapper import *

from data import Vocab
from model import SummarizationModel
from batcher import Batcher

import xml_parser
import convnet

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

def build_batch(pair_list, vocab, hp):
    instance_list=[batcher.Example(article=article, abstract_sentences=abstract, vocab=vocab, hps=hp)
        for article, abstract in pair_list]
    batch=batcher.Batch(example_list=instance_list, hps=hp, vocab=vocab)
    return batch

def train_model(ext_solver, abs_model, data_manager):
    train_dir=os.path.join(FLAGS.log_root, 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    with tf.device('/cpu:0'):
        abs_model.build_graph()
        saver=tf.train.Saver(max_to_keep=100000)        # save a lot of check point

    sv=tf.train.Supervisor(logdir=train_dir,
        is_chief=True,
        saver=saver,
        summary_op=None,
        save_summaries_secs=3600,       # save summaries for tensorboard every hour
        save_model_secs=3600,           # check point saved every hour
        global_step=abs_model.global_step)
    summary_writer=sv.summary_writer
    tf.logging.info('Preparing or waiting for session ...')
    sess_context_manager=sv.prepare_or_wait_for_session(config=util.get_config())
    tf.logging.info('Created session.')

    batch_size=abs_model._hps.batch_size

    try:
        batch_idx=0
        pair_list=[]
        loss_dict={}
        loss_list=[]
        while True:
            shuffle(data_manager.text_abstract_pair)
            for idx, (article, abstract) in enumerate(data_manager.text_abstract_pair):
                article=article.decode('ascii', errors='ignore').encode('ascii')
                abstract=[item.decode('ascii', errors='ignore').encode('ascii') for item in abstract]

                # Extractive model
                sentence_list=tokenize.sent_tokenize(article)
                tokenized_article='\n'.join(sentence_list)
                with open('tmp.txt','w') as fopen:
                    fopen.write(tokenized_article)
                selected_sentence=ext_solver.select('tmp.txt')
                extracted_content=selected_sentence.replace('\n',' ').lower()
                pair_list.append((extracted_content, abstract))

                # Train abstractive model
                if len(pair_list)==batch_size:
                    batch_idx+=1
                    this_batch=build_batch(pair_list=pair_list, vocab=abs_model._vocab, hp=abs_model._hps)
                    results=abs_model.run_train_step(sess_context_manager, this_batch)
                    loss=results['loss']
                    if FLAGS.coverage:
                        coverage_loss=results['coverage_loss']
                        sys.stdout.write('batch_idx = %d, loss = %.4f, coverage_loss = %.4f\r'%(
                            batch_idx, loss, coverage_loss))
                    else:
                        sys.stdout.write('batch_idx = %d, loss = %.4f\r'%(
                            batch_idx, loss))
                    sys.stdout.flush()
                    loss_list.append(loss)

                    if batch_idx%100==0:
                        loss_dict[batch_idx]=np.mean(loss_list)
                        loss_list=[]
                        print('Average loss between [%d, %d) = %.4f'%(batch_idx-100, batch_idx, loss_dict[batch_idx]))

                    # Reset
                    pair_list=[]
    except:
        print('')
        print('Received a break signal from the user')
        traceback.print_exc()


def main(unused_argv):
    if len(unused_argv)!=1:
        raise Exception('Problem with flags: %s'%str(unused_argv))

    try:
        assert(FLAGS.mode=='train')
    except:
        raise ValueError('mode must be "train" while now it is "%s"'%FLAGS.mode)

    FLAGS.log_root=os.path.join(FLAGS.log_root, FLAGS.exp_name)
    data_manager=BinaryDataManager(binary_file=FLAGS.data_path, single_pass=True)
    data_manager.load_data()

    model_hp_list=['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
        'hidden_dim', 'emb_dim', 'batch_size', 'max_enc_steps', 'max_dec_steps', 'coverage', 'cov_loss_wt', 'pointer_gen']
    model_hp_dict={}
    for key,value in FLAGS.__flags.iteritems():
        if key in model_hp_list:
            model_hp_dict[key]=value
    model_settings=namedtuple('HParams',model_hp_dict.keys())(**model_hp_dict)
    model_settings=model_settings._replace(max_dec_steps=1)

    vocab=Vocab(FLAGS.vocab_path, FLAGS.vocab_size)

    # Lauch extractive model
    cur_path=os.path.abspath('.')
    FLAGS.sentence_extract_config=os.path.abspath(FLAGS.sentence_extract_config)
    os.chdir(FLAGS.sentence_extract_root)
    sys.path.insert(0, 'run')
    sys.path.insert(0, 'util')
    import laucher
    import xml_parser
    laucher_params=xml_parser.parse(FLAGS.sentence_extract_config, flat=False)
    ext_solver=laucher.laucher(laucher_params)
    ext_solver.start()
    os.chdir(cur_path)

    # Launch abstractive model
    loaded_params=tf.global_variables()
    abs_model=SummarizationModel(model_settings, vocab, extra_info={})
    train_model(ext_solver=ext_solver, abs_model=abs_model, data_manager=data_manager)

if __name__=='__main__':
    tf.app.run()
