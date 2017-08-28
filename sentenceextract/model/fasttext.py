import sys
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
sys.path.insert(0,'./util')
from py2py3 import *
import numpy as np
import tensorflow as tf

sys.path.insert(0,'./model')
from __init__ import *

class fastText(base_net):

    def __init__(self, hyper_params):
        '''
        >>> Construct a fastText model
        >>> hyper_params: dict, a dictionary containing all hyper parameters
            >>> batch_size: int, batch size
            >>> sequence_length: int, maximum sentence length
            >>> class_num: int, number of categories
            >>> vocab_size: int, vocabulary size
            >>> embedding_dim: int, dimension of word embeddings
            >>> update_policy: dict, update policy
            >>> embedding_matrix: optional, numpy.array, initial embedding matrix
            >>> embedding_trainable: optional, bool, whether or not the embedding is trainable, default is true
        '''
        self.batch_size=hyper_params['batch_size']
        self.sequence_length=hyper_params['sequence_length']
        self.class_num=hyper_params['class_num']
        self.vocab_size=hyper_params['vocab_size']
        self.embedding_dim=hyper_params['embedding_dim']
        self.update_policy=hyper_params['update_policy']
        self.embedding_trainable=hyper_params['embedding_trainable'] if 'embedding_trainable' in hyper_params else True
        self.grad_clip_norm=hyper_params['grad_clip_norm'] if 'grad_clip_norm' in hyper_params else 1.0
        self.name='fast_text model' if not 'name' in hyper_params else hyper_params['name']

        self.sess=None
        with tf.variable_scope('fastText'):
            if not 'embedding_matrix' in hyper_params:
                print('Word embeddings are initialized from scrach')
                self.embedding_matrix=tf.get_variable('embedding_matrix', shape=[self.vocab_size, self.embedding_dim],
                    initializer=tf.random_uniform_initializer(-1.0,1.0), dtype=tf.float32)
            else:
                print('Pre-trained word embeddings are imported')
                embedding_value=tf.constant(hyper_params['embedding_matrix'], dtype=tf.float32)
                self.embedding_matrix=tf.get_variable('embedding_matrix', initializer=embedding_value, dtype=tf.float32)

        self.inputs=tf.placeholder(tf.int32,shape=[self.batch_size, self.sequence_length])
        self.masks=tf.placeholder(tf.int32,shape=[self.batch_size, self.sequence_length])
        self.labels=tf.placeholder(tf.int32,shape=[self.batch_size,])

        self.embedding_output=tf.nn.embedding_lookup(self.embedding_matrix,self.inputs)             # of shape [self.batch_size, self.sequence_length, self.embedding_dim]
        embedding_sum=tf.reduce_sum(self.embedding_output,axis=1)                                   # of shape [self.batch_size, self.embedding_dim]
        mask_sum=tf.reduce_sum(self.masks,axis=1)                                                   # of shape [self.batch_size,]
        mask_sum=tf.expand_dims(mask_sum,axis=-1)                                                   # of shape [self.batch_size, 1]

        self.sentence_embedding=tf.div(embedding_sum, tf.cast(mask_sum, dtype=tf.float32))                                     # broadcasting mask_sum, the embedding of padded token must be zero

        # Construct softmax classifier
        with tf.variable_scope('fastText'):
            W=tf.get_variable(name='w',shape=[self.embedding_dim,self.class_num],
                initializer=tf.truncated_normal_initializer(stddev=0.5))
            b=tf.get_variable(name='b',shape=[self.class_num,],
                initializer=tf.truncated_normal_initializer(stddev=0.05))
        output=tf.add(tf.matmul(self.sentence_embedding,W),b)                                       # of shape [self.batch_size, self.class_num], unnormalized output probability distribution

        # Outputs
        self.probability=tf.nn.softmax(output)                                                      # of shape [self.batch_size, self.class_num], normalized output probability distribution
        self.prediction=tf.argmax(self.probability,axis=1)                                          # of shape [self.batch_size], discrete prediction
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,labels=self.labels))  # loss value

        # Construct Optimizer
        if self.update_policy['name'].lower() in ['sgd', 'stochastic gradient descent']:
            learning_rate=self.update_policy['learning_rate']
            momentum=0.0 if not 'momentum' in self.update_policy else self.update_policy['momentum']
            self.optimizer=tf.train.MomentumOptimizer(learning_rate, momentum)
        elif self.update_policy['name'].lower() in ['adagrad',]:
            learning_rate=self.update_policy['learning_rate']
            initial_accumulator_value=0.1 if not 'initial_accumulator_value' in self.update_policy \
                else self.update_policy['initial_accumulator_value']
            self.optimizer=tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value)
        elif self.update_policy['name'].lower() in ['adadelta']:
            learning_rate=self.update_policy['learning_rate']
            rho=0.95 if not 'rho' in self.update_policy else self.update_policy['rho']
            epsilon=1e-8 if not 'epsilon' in self.update_policy else self.update_policy['epsilon']
            self.optimizer=tf.train.AdadeltaOptimizer(learning_rate, rho, epsilon)
        elif self.update_policy['name'].lower() in ['rms', 'rmsprop']:
            learning_rate=self.update_policy['learning_rate']
            decay=0.9 if not 'decay' in self.update_policy else self.update_policy['decay']
            momentum=0.0 if not 'momentum' in self.update_policy else self.update_policy['momentum']
            epsilon=1e-10 if not 'epsilon' in self.update_policy else self.update_policy['epsilon']
            self.optimizer=tf.train.RMSPropOptimizer(learning_rate, decay, momentum, epsilon)
        elif self.update_policy['name'].lower() in ['adam']:
            learning_rate=self.update_policy['learning_rate']
            beta1=0.9 if not 'beta1' in self.update_policy else self.update_policy['beta1']
            beta2=0.999 if not 'beta2' in self.update_policy else self.update_policy['beta2']
            epsilon=1e-8 if not 'epsilon' in self.update_policy else self.update_policy['epsilon']
            self.optimizer=tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon)
        else:
            raise ValueError('Unrecognized Optimizer Category: %s'%self.update_policy['name'])

        print('gradient clip is applied, max = %.2f'%self.grad_clip_norm)
        gradients=self.optimizer.compute_gradients(self.loss)
        clipped_gradients=[(tf.clip_by_value(grad,-self.grad_clip_norm,self.grad_clip_norm),var) for grad,var in gradients]
        self.update=self.optimizer.apply_gradients(clipped_gradients)

    def train_validate_test_init(self):
        '''
        >>> Initialize the training validation and test phrase
        '''
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self,inputs,masks,labels):
        '''
        >>> Training phrase
        '''
        train_dict={self.inputs:inputs,self.masks:masks,self.labels:labels}
        self.sess.run(self.update,feed_dict=train_dict)
        prediction_this_batch, loss_this_batch=self.sess.run([self.prediction,self.loss],feed_dict=train_dict)
        return prediction_this_batch, loss_this_batch

    def validate(self,inputs,masks,labels):
        '''
        >>> Validation phrase
        '''
        validate_dict={self.inputs:inputs,self.masks:masks,self.labels:labels}
        prediction_this_batch, loss_this_batch=self.sess.run([self.prediction,self.loss],feed_dict=validate_dict)
        return prediction_this_batch, loss_this_batch

    def test(self,inputs,masks,fine_tune=False):
        '''
        >>> Test phrase
        '''
        test_dict={self.inputs:inputs,self.masks:masks}
        if fine_tune==False:
            prediction_this_batch,=self.sess.run([self.prediction,],feed_dict=test_dict)
        else:
            prediction_this_batch,=self.sess.run([self.probability,],feed_dict=test_dict)
        return prediction_this_batch

    def do_summarization(self, file_list, folder2store, data_generator, n_top=5):
        '''
        >>> Not implemented
        '''
        raise NotImplementedError('Current model is fasttext, where "do_summarization" function is not implemented')

    def dump_params(self,file2dump):
        '''
        >>> Save the parameters
        >>> file2dump: str, file to store the parameters
        '''
        saver=tf.train.Saver()
        saved_path=saver.save(self.sess, file2dump)
        print('parameters are saved in file %s'%saved_path)

    def load_params(self,file2load, loaded_params=[]):
        '''
        >>> Load the parameters
        >>> file2load: str, file to load the parameters
        '''
        param2load=[]
        for var in tf.global_variables():
            if not var in loaded_params:
                param2load.append(var)
        saver=tf.train.Saver(param2load)
        saver.restore(self.sess, file2load)
        print('parameters are imported from file %s'%file2load)

    def train_validate_test_end(self):
        '''
        >>> End current training validation and test phrase
        '''
        self.sess.close()
        self.sess=None

