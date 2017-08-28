import os
import sys
import numpy as np
import tensorflow as tf
from collections import namedtuple
if sys.version_info.major==2:
    import cPickle
else:
    import pickle as cPickle

file_path=os.path.dirname(os.path.realpath(__file__))
file_path=os.path.join(os.path.join(file_path,os.pardir),'model')
sys.path.insert(0, file_path)

import util
from data import Vocab
from batcher import Batcher
from model import SummarizationModel
from decode import BeamSearchDecoder

FLAGS=tf.app.flags.FLAGS

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
tf.app.flags.DEFINE_boolean('coverage', True, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')

# check points listed
tf.app.flags.DEFINE_integer('max_ckpt_num', 10, 'The maximum checkpoint to be evaluated')
tf.app.flags.DEFINE_integer('interval', 5, 'The interval between two neighboring evaluated points')

def get_config():
    config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))
    config.gpu_options.allow_growth=True
    return config

def get_ckpt_list(folder, max_ckpt_num, interval):
    ckpt_list=[]
    for file in os.listdir(folder):
        if file.split('.')[-1] in ['meta']:
            pure_name='.'.join(file.split('.')[:-1])
            batch_idx=int(pure_name.split('-')[-1])
            ckpt_list.append((pure_name, batch_idx))
    if sys.version_info.major==2:
        ckpt_list=sorted(ckpt_list, lambda x,y: 1 if x[1]<y[1] else -1)
    else:
        ckpt_list=sorted(ckpt_list, key=lambda x:x[1], reverse=True)

    interval=max(1, interval)
    selected_ckpt_list=[ckpt_list[idx][0] for idx in np.arange(0, len(ckpt_list), interval)]

    if max_ckpt_num>0:
        return selected_ckpt_list[:max_ckpt_num]
    else:
        return selected_ckpt_list

def load_ckpt(saver, sess, ckpt):
    saver=tf.train.Saver() if saver==None else saver
    saved_path=saver.restore(sess, ckpt)
    return saved_path

def eval(model,batcher,vocab,sess):
    loss_list=[]
    while True:
        batch=batcher.next_batch()
        if batch==None:
            break

        results=model.run_eval_step(sess,batch)
        loss=results['loss']
        if FLAGS.coverage:
            coverage_loss=results['coverage_loss']
            loss+=coverage_loss
        loss_list.append(loss)
        sys.stdout.write('Processing %d batches, average loss=%.3f\r'%(len(loss_list),np.mean(loss_list)))
    return np.mean(loss_list)

def main(unused_argv):
    if len(unused_argv)!=1:
        raise Exception('Problem with flags: %s'%unused_argv)

    FLAGS.log_root=os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        raise Exception('log directory %s does not exist.'%FLAGS.log_root)

    vocab=Vocab(FLAGS.vocab_path, FLAGS.vocab_size)

    hparam_list=['mode','lr','adagrad_init_acc','rand_unif_init_mag','trunc_norm_init_std','max_grad_norm','hidden_dim','emb_dim','batch_size','max_dec_steps','max_enc_steps','coverage','cov_loss_wt','pointer_gen']
    hps_dict={}
    for key,val in FLAGS.__flags.iteritems(): # for each flag
        if key in hparam_list: # if it's in the list
            hps_dict[key]=val # add it to the dict
    hps=namedtuple("HParams", hps_dict.keys())(**hps_dict)    

    model=SummarizationModel(hps,vocab)

    result_map=[]
    model.build_graph()
    sess=tf.Session(config=get_config())
    trained_model_folder=os.path.join(FLAGS.log_root,'train')

    evaluation_folder=os.path.join(FLAGS.log_root,'eval')
    ckpt_list=get_ckpt_list(trained_model_folder, max_ckpt_num=FLAGS.max_ckpt_num, interval=FLAGS.interval)
    if os.path.exists(evaluation_folder+os.sep+'result.pkl'):
        result_map=cPickle.load(open(evaluation_folder+os.sep+'result.pkl','rb'))
        ckpt_list_included=[]
        ckpt_list_extra=[]
        for ckpt_file, loss in result_map:
            ckpt_list_included.append(ckpt_file)
        for ckpt_file in ckpt_list:
            if not ckpt_file in ckpt_list_included:
                ckpt_list_extra.append(ckpt_file)
        ckpt_list=ckpt_list_extra
        print('%d ckpt already included in the existing result.pkl, skip ...'%len(ckpt_list_included))
    print('There are %d ckpts to evaluate'%len(ckpt_list))

    for idx,ckpt_file in enumerate(ckpt_list):
        print('Start analyzing checkpoint %d/%d'%(idx+1,len(ckpt_list)))
        saver=tf.train.Saver(max_to_keep=3)
        load_ckpt(saver,sess,os.path.join(trained_model_folder,ckpt_file))
        batcher=Batcher(FLAGS.data_path,vocab,hps,single_pass=True)
        avg_loss=eval(model,batcher,vocab,sess)
        print('check point:%s, Average loss in validation set: %.3f'%(ckpt_file, avg_loss))
        result_map.append([ckpt_file,avg_loss])
        if not os.path.exists(evaluation_folder):
            os.makedirs(evaluation_folder)
        cPickle.dump(result_map,open(evaluation_folder+os.sep+'result.pkl','wb'))

    if sys.version_info.major==2:
        result_map=sorted(result_map,lambda x,y:-1 if x[1]>y[1] else 1)
    else:
        result_map=sorted(result_map,key=lambda x:x[1],reverse=True)
    print('==Summary==')
    for ckpt,avg_loss in result_map:
        print('check point: %s, average loss: %.3f'%(ckpt,avg_loss))
    cPickle.dump(result_map,open(evaluation_folder+os.sep+'result.pkl','wb'))
    print('results saved in %s'%(evaluation_folder+os.sep+'result.pkl'))

if __name__=='__main__':
    tf.app.run()

