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

file_path=os.path.dirname(os.path.realpath(__file__))
file_path=os.path.join(os.path.join(file_path,os.pardir),'model')
sys.path.insert(0, file_path)

import util
import data
import model
import batcher
import beam_search

from data_wrapper import *
from model_wrapper import *

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

# Saving folders
tf.app.flags.DEFINE_string('article_folder','','folder that saves the article')
tf.app.flags.DEFINE_string('refer_folder','','folder that saves the reference summary')
tf.app.flags.DEFINE_string('output_folder','','folder that saves the output summary')

def main(unused_argv):
    if len(unused_argv)!=1:
        raise Exception('Problem with flags: %s'%str(unused_argv))

    try:
        assert(FLAGS.mode=='decode')
    except:
        raise ValueError('mode must be "decode" while now it is "%s"'%FLAGS.mode)

    FLAGS.log_root=os.path.join(FLAGS.log_root, FLAGS.exp_name)
    try:
        assert(os.path.exists(FLAGS.log_root))
    except:
        raise ValueError('invalid log_root: %s, file or folder not exists'%str(FLAGS.log_root))
    FLAGS.batch_size=FLAGS.beam_size

    data_manager=BinaryDataManager(binary_file=FLAGS.data_path, single_pass=True)
    data_manager.load_data()

    model_hp_list=['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
        'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt', 'pointer_gen']
    model_hp_dict={}
    for key,value in FLAGS.__flags.iteritems():
        if key in model_hp_list:
            model_hp_dict[key]=value
    model_settings=namedtuple('HParams',model_hp_dict.keys())(**model_hp_dict)
    model_settings=model_settings._replace(max_dec_steps=1)

    for folder in [FLAGS.article_folder, FLAGS.refer_folder, FLAGS.output_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Launch extractive model
    cur_path=os.path.abspath('.')
    FLAGS.sentence_extract_config=os.path.abspath(FLAGS.sentence_extract_config)
    os.chdir(FLAGS.sentence_extract_root)
    sys.path.insert(0,'run')
    sys.path.insert(0,'util')
    import laucher
    import xml_parser
    laucher_params=xml_parser.parse(FLAGS.sentence_extract_config, flat=False)
    ext_solver=laucher.laucher(laucher_params)
    ext_solver.start()
    os.chdir(cur_path)

    # Launch abstractive model
    loaded_params=tf.global_variables()
    abs_solver=RunTimeWrapper(hp=FLAGS, model_settings=model_settings)
    abs_solver.start(loaded_params=loaded_params)

    # Preparation
    for folder in [FLAGS.article_folder, FLAGS.refer_folder, FLAGS.output_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    result2write=''
    for idx, (article, abstract) in enumerate(data_manager.text_abstract_pair):
        sys.stdout.write('Analysizing the documents %d/%d = %.1f %%\r'%(idx+1, len(data_manager.text_abstract_pair),
            float(idx+1)/float(len(data_manager.text_abstract_pair))*100))
        sys.stdout.flush()
        try:
            article=article.decode('ascii', errors='ignore').encode('ascii')
            sentence_list=tokenize.sent_tokenize(article)
            tokenized_article='\n'.join(sentence_list)
            with open('tmp.txt','w') as fopen:
                fopen.write(tokenized_article)
            selected_sentence=ext_solver.select('tmp.txt')
            extracted_content=selected_sentence.replace('\n', ' ').lower()
            _, summary=abs_solver.run(query=extracted_content)
            # Reference and compare
            with open(FLAGS.article_folder+os.sep+'%04d_article.txt'%idx,'w') as fopen:
                fopen.write(article)
            with open(FLAGS.refer_folder+os.sep+'%04d_reference.txt'%idx,'w') as fopen:
                fopen.write(abstract)
            with open(FLAGS.output_folder+os.sep+'%04d_decode.txt'%idx,'w') as fopen:
                fopen.write(summary)
            result2write+='\n\n===\n%s\n\n>>>refer:\n%s\n\n>>>output:\n%s\n'%(article,abstract,summary)
            if (idx+1)%100==0:
                with open('results.txt','w') as fopen:
                    fopen.write(result2write)
        except:
            print('ERROR while loading document %d'%idx)
            traceback.print_exc()

    ext_solver.end()
    abs_solver.end()

if __name__=='__main__':
    tf.app.run()



