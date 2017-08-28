import os
import sys
sys.path.insert(0,'./util')
sys.path.insert(0,'./model')
import numpy as np
import tensorflow as tf
if sys.version_info.major==2:
    import cPickle
else:
    import pickle as cPickle

from py2py3 import *

class PGNet(object):

    def __init__(self, hyper_params):
        '''
        >>> hyper_params: dict, containing all necessary hyper_params
        '''

        self.name=hyper_params['name']
        self.mode=hyper_params['mode']
        self.batch_size=hyper_params['batch_size']
        self.vocab_size=hyper_params['vocab_size']
        self.embedding_dim=hyper_params['embedding_dim']
        self.encoding_dim=hyper_params['encoding_dim']
        self.decoding_dim=hyper_params['decoding_dim']
        self.max_encoding_step=hyper_params['max_encoding_step']
        self.max_decoding_step=hyper_params['max_decoding_step']
        self.attention_mode=hyper_params['attention_mode'] if 'attention_mode' in hyper_params else 'linear'
        self.update_policy=hyper_params['update_policy']

        # For sent2vec
        self.sent2vec_lambda=0.2 if not 'sent2vec_lambda' in hyper_params else hyper_params['sent2vec_lambda']

        if self.mode in ['decode',] and self.max_decoding_step!=1:
            raise ValueError('For %s mode, the max_decoding_step must be 1'%self.mode)

        self.use_pointer=False if not 'use_pointer' in hyper_params else hyper_params['use_pointer']
        self.loss_type='max_likelihood' if not 'loss_type' in hyper_params else hyper_params['loss_type']
        self.grad_clip_norm=1.0 if not 'grad_clip_norm' in hyper_params else hyper_params['grad_clip_norm']

        self.sess=None
        self.saver=None
        self.sub_network={}

        self.encode_input_batch=tf.placeholder(tf.int32, [self.batch_size, self.max_encoding_step], name='encode_input_batch')
        self.encode_input_length=tf.placeholder(tf.int32, [self.batch_size], name='encode_input_length')
        self.decode_input_batch=tf.placeholder(tf.int32, [self.batch_size, self.max_decoding_step], name='decode_input_batch')
        self.decode_ground_truth=tf.placeholder(tf.int32, [self.batch_size, self.max_decoding_step], name='decode_ground_truth')
        self.decode_mask=tf.placeholder(tf.int32, [self.batch_size, self.max_decoding_step], name='decode_mask')

        if 'pretrained_embeddings' in hyper_params and hyper_params['pretrained_embeddings']==True:
            if 'embedding_matrix' not in hyper_params:
                raise ValueError('Try to import pretrained embeddings but not valid embeddings are found')
            self.embedding_matrix=hyper_params['embedding_matrix']
            if not tuple(self.embedding_matrix.shape)==(self.vocab_size, self.embedding_dim):
                raise ValueError('The embedding matrix to import is of wrong shape. %s expected but %s found'%(
                    str(tuple((self.vocab_size, self.embedding_dim))),str(self.embedding_matrix.shape)))
        else:
            self.embedding_matrix=None

        self.build_encoder()
        self.build_decoder()
        if self.loss_type=='sent2vec':
            assert 'convnet' in hyper_params, 'loss_type=sent2vec, however no valid sent2vec network is found'
            self.load_subnetwork(network=hyper_params['convnet'], name='sent2vec')

        self.build_classifier()

    def load_subnetwork(self, network, name):
        if name.lower() in self.sub_network:
            raise ValueError('sub network %s has already been loaded!'%(name))

        if name.lower() in ['sent2vec',]:
            assert network.batch_size==self.batch_size
            assert network.embedding_dim==self.embedding_dim
            assert network.vocab_size==self.vocab_size
            self.sub_network['sent2vec']=network
        else:
            raise ValueError('Unrecognized name of sub network: %s'%name)

    def build_encoder(self,):
        # Embedding matrix initialization
        if self.embedding_matrix==None:
            self.embeddings=tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0), dtype=tf.float32)
            print('Embedding matrix of shape (%d, %d) is initialized from scratch'%(self.vocab_size, self.embedding_dim))
        else:
            self.embeddings=tf.Variable(self.embedding_matrix,dtype=tf.float32,name='embeddings')
            print('Embedding matrix of shape (%d, %d) is imported into the model'%(self.vocab_size, self.embedding_dim))

        self.encode_embedding_output=tf.nn.embedding_lookup(self.embeddings, self.encode_input_batch)       # of shape [self.batch_size, self.max_encoding_step, self.embedding_dim]

        # Encoding Sequence
        with tf.variable_scope('encoder'):
            cell_fw=tf.contrib.rnn.LSTMCell(self.encoding_dim, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000), state_is_tuple=True)
            cell_bw=tf.contrib.rnn.LSTMCell(self.encoding_dim, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000), state_is_tuple=True)
            self.encoding_states, (fw_state, bw_state)=tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.encode_embedding_output,
                dtype=tf.float32, sequence_length=self.encode_input_length, swap_memory=True)
            self.encoding_states=tf.concat(axis=2, values=self.encoding_states)     # of shape [self.batch_size, self.max_encoding_step, 2*self.encoding_dim]

        # Combine the Two Final States
        with tf.variable_scope('reduce'):
            Wc=tf.get_variable('Wc_reduce', [self.encoding_dim*2, self.encoding_dim], dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000))
            Wh=tf.get_variable('Wh_reduce', [self.encoding_dim*2, self.encoding_dim], dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000))
            bc=tf.get_variable('bc_reduce', [self.encoding_dim,], dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000))
            bh=tf.get_variable('bh_reduce', [self.encoding_dim,], dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000))

            ori_c=tf.concat(axis=1, values=[fw_state.c, bw_state.c])            # of shape [self.batch_size, 2*self.encoding_dim]
            ori_h=tf.concat(axis=1, values=[fw_state.h, bw_state.h])            # of shape [self.batch_size, 2*self.encoding_dim]
            prj_c=tf.nn.relu(tf.matmul(ori_c, Wc)+bc)                           # of shape [self.batch_size, self.encoding_dim]
            prj_h=tf.nn.relu(tf.matmul(ori_h, Wh)+bh)                           # of shape [self.batch_size, self.encoding_dim]
            self.init_decode_state=tf.contrib.rnn.LSTMStateTuple(prj_c, prj_h)

    def build_decoder(self,):
        self.decode_embedding_output=tf.nn.embedding_lookup(self.embeddings, self.decode_input_batch)       # of shape [self.batch_size, self.max_decoding_step, self.embedding_dim]
        self.decode_inputs=tf.split(self.decode_embedding_output, self.max_decoding_step, axis=1)           # list of tensors of shape [self.batch_size, 1, self.embedding_dim]

        # For Debug
        self.debug_decode_representation=[]
        self.W_out=None
        self.b_out=None
        # End of Debug field
        prediction_sequence=[]                                                                              # list of output prediction (normalized)
        with tf.variable_scope('decoder'):
            cell_decode=tf.contrib.rnn.LSTMCell(self.decoding_dim, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000), state_is_tuple=True)

            self.encode_att_init=tf.get_variable('encode_att_init', dtype=tf.float32, initializer=tf.constant(np.ones([self.batch_size, self.max_encoding_step], dtype=np.float32)), trainable=False)

            self.encode_att_history=[]       # list of all encode attention histories
            self.encode_att_history.append(self.encode_att_init)

            self.lstm_decode_state=self.init_decode_state
            self.init_decode_hidden_states=tf.placeholder(tf.float32, shape=[self.batch_size, None, self.decoding_dim], name='init_decode_hidden_states')   # is non-empty only in the decoding phrase
            output_sequence=[self.init_decode_hidden_states,]

            for decode_idx, decode_input_token in enumerate(self.decode_inputs):
                if decode_idx>0:
                    tf.get_variable_scope().reuse_variables()

                decode_input_token=tf.reshape(decode_input_token, [self.batch_size, self.embedding_dim])    # of shape [self.batch_size, self.embedding_dim]
                output, self.lstm_decode_state=cell_decode(decode_input_token, self.lstm_decode_state)

                # Compress the LSTM states to fit the W_encode_att, W_decode_att
                h_decode_states_2d=self.project(variables=self.lstm_decode_state, output_size=self.decoding_dim, bias=0.0, scope_name='LSTM_decode_projection') # of shape [self.batch_size, self.decoding_dim]
                self.h_decode_states=tf.expand_dims(h_decode_states_2d, axis=-1)    # of shape [self.batch_size, self.decoding_dim, 1]
                output_sequence.append(tf.reshape(self.h_decode_states,[self.batch_size, 1, self.decoding_dim]))
                self.decoding_states=tf.concat(output_sequence,axis=1)  # of shape [self.batch_size, decoding_states_num, self.decoding_dim]

                if self.attention_mode in ['linear']:
                    print('Attention Calculation Mode: linear')
                    # To calculate the encoding attention
                    self.s_encode_att=tf.get_variable('s_encode_att', [self.encoding_dim, self.decoding_dim], dtype=tf.float32,
                                                      initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000))
                    self.h_encode_att=tf.get_variable('h_encode_att', [self.encoding_dim, self.encoding_dim * 2], dtype=tf.float32,
                                                      initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000))
                    self.v_encode_att=tf.get_variable('v_encode_att', [self.encoding_dim,], dtype=tf.float32,
                                                      initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000))

                    self.s_encode_att=tf.tile(tf.reshape(self.s_encode_att, [1, self.encoding_dim, self.decoding_dim]),
                                              multiples=[self.batch_size, 1, 1])        # of shape [self.batch_size, self.encoding_dim, self.decoding_dim]
                    self.h_encode_att=tf.tile(tf.reshape(self.h_encode_att, [1, self.encoding_dim, self.encoding_dim * 2]),
                                              multiples=[self.batch_size, 1, 1])        # of shape [self.batch_size, self.encoding_dim, self.encoding_dim * 2]
                    self.v_encode_att=tf.tile(tf.reshape(self.v_encode_att, [1, 1, self.encoding_dim]),
                                              multiples=[self.batch_size, 1, 1])        # of shape [self.batch_size, 1, self.encoding_dim]

                    decode_part_encode_att=tf.tile(tf.matmul(self.s_encode_att, self.h_decode_states), multiples=[1, 1, self.max_encoding_step])              # of shape [self.batch_size, self.encoding_dim, self.max_encoding_step]
                    encode_part_encode_att=tf.matmul(self.h_encode_att, tf.transpose(self.encoding_states, perm=[0, 2, 1]))                                   # of shape [self.batch_size, self.encoding_dim, self.max_encoding_step]

                    self.encode_att=tf.matmul(self.v_encode_att, tf.tanh(decode_part_encode_att+encode_part_encode_att))         # of shape [self.batch_size, 1, self.max_encoding_step]
                    self.encode_att=tf.reshape(self.encode_att, [self.batch_size, self.max_encoding_step])

                    # To calculate the decoding attention
                    self.s_decode_att=tf.get_variable('s_decode_att', [self.decoding_dim, self.decoding_dim], dtype=tf.float32,
                                                      initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000))
                    self.h_decode_att=tf.get_variable('h_decode_att', [self.decoding_dim, self.decoding_dim], dtype=tf.float32,
                                                      initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000))
                    self.v_decode_att=tf.get_variable('v_decode_att', [self.decoding_dim,], dtype=tf.float32,
                                                      initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000))

                    self.s_decode_att=tf.tile(tf.reshape(self.s_decode_att, [1, self.decoding_dim, self.decoding_dim]),
                                              multiples=[self.batch_size, 1, 1])        # of shape [self.batch_size, self.decoding_dim, self.decoding_dim]
                    self.h_decode_att=tf.tile(tf.reshape(self.h_decode_att, [1, self.decoding_dim, self.decoding_dim]),
                                              multiples=[self.batch_size, 1, 1])        # of shape [self.batch_size, self.decoding_dim, self.decoding_dim]
                    self.v_decode_att=tf.tile(tf.reshape(self.v_decode_att, [1, 1, self.decoding_dim]),
                                              multiples=[self.batch_size, 1, 1])        # of shape [self.batch_size, 1, self.decoding_dim]

                    current_part_decode_att=tf.matmul(self.s_decode_att, self.h_decode_states)                                                 # of shape [self.batch_size, self.decoding_dim, 1]
                    history_part_decode_att=tf.matmul(self.h_decode_att, tf.transpose(self.decoding_states, perm=[0, 2, 1]))                   # of shape [self.batch_size, self.decoding_dim, decoding_states_num]

                    self.decode_att=tf.matmul(self.v_decode_att, tf.tanh(current_part_decode_att+history_part_decode_att))                     # of shape [self.batch_size, 1, decoding_states_num] #BROADCAST
                    self.decode_att=tf.reshape(self.decode_att, [self.batch_size, -1])

                elif self.attention_mode in ['multiply']:
                    print('Attention Calculation Mode: multiply')
                    # To calculate the encoding attention
                    self.W_encode_att=tf.get_variable('W_encode_att', [self.encoding_dim * 2, self.decoding_dim], dtype=tf.float32,
                                                      initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000))

                    # Broadcast the matrix for batch matrix multiplication
                    self.W_encode_att=tf.tile(tf.reshape(self.W_encode_att, [1, self.encoding_dim * 2, self.decoding_dim]),
                        multiples=[self.batch_size, 1, 1])  # of shape [self.batch_size, self.encoding_dim*2, self.decoding_dim]
                    self.W_decode_att=tf.tile(tf.reshape(self.W_decode_att, [1, self.decoding_dim, self.decoding_dim]),
                        multiples=[self.batch_size, 1, 1])  # of shape [self.batch_size, self.decoding_dim, self.decoding_dim]

                    # intra-attention in encoding part
                    self.encode_att=tf.matmul(tf.matmul(self.encoding_states, self.W_encode_att), self.h_decode_states)      # of shape [self.batch_size,  self.max_encoding_step, 1]
                    self.encode_att=tf.reshape(self.encode_att, [self.batch_size, self.max_encoding_step])

                    # To calculate the decoding attention
                    self.W_decode_att=tf.get_variable('W_decode_att', [self.decoding_dim, self.decoding_dim], dtype=tf.float32,
                                                      initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000))
                    self.decode_att=tf.matmul(tf.matmul(self.decoding_states, self.W_decode_att), self.h_decode_states)                      # of shape [self.batch_size, #Decoded States, 1]
                else:
                    raise ValueError('Unrecognized attention mode: %s'%self.attention_mode)

                # Normalize the intra-encoding attention
                encode_att_prime=tf.div(tf.exp(self.encode_att), self.encode_att_history[-1])                            # of shape [self.batch_size, self.max_encoding_step]
                self.encode_att_history.append(self.encode_att_history[-1] + tf.exp(self.encode_att))                    # update the encoding attention history
                self.encode_att=tf.nn.softmax(encode_att_prime, dim=-1)                                                  # final attention of shape [self.batch_size, self.max_encoding_step]
                self.encode_att_vector=tf.matmul(tf.reshape(self.encode_att,[self.batch_size, 1, self.max_encoding_step]), self.encoding_states)  # of shape [self.batch_size, 1, self.encoding_dim*2]
                self.encode_att_vector=tf.reshape(self.encode_att_vector,[self.batch_size, 2*self.encoding_dim])         # of shape [self.batch_size, 2*self.max_encoding_step]

                # Normalize the intra-decoding attention
                self.decode_att=tf.nn.softmax(tf.reshape(self.decode_att, [self.batch_size, -1]), dim=-1)                                # of shape [self.batch_size, decoding_states_num]
                self.decode_att_vector=tf.matmul(tf.reshape(self.decode_att, [self.batch_size, 1, -1]),  self.decoding_states)           # of shape [self.batch_size, 1, self.decoding_dim]
                self.decode_att_vector=tf.reshape(self.decode_att_vector, [self.batch_size, self.decoding_dim])                          # of shape [self.batch_size, self.decoding_dim]

                decode_representation=tf.concat([h_decode_states_2d, self.encode_att_vector, self.decode_att_vector], axis=1)            # of shape [self.batch_size, (self.decoding_dim*2 + self.encoding_dim*2)]

                W_output=tf.get_variable('W_output', [self.decoding_dim*2+self.encoding_dim*2, self.vocab_size], dtype=tf.float32,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000))
                b_output=tf.get_variable('b_output', [self.vocab_size,], dtype=tf.float32, initializer=tf.random_uniform_initializer(-1.0, 1.0, seed=10000))
                self.prediction=tf.matmul(decode_representation,W_output)+b_output                                                       # of shape [self.batch_size, self.vocab_size]

                ## For Debug
                self.debug_decode_representation.append(decode_representation)
                self.W_out=W_output
                self.b_out=b_output
                ## End of Debug

                # if we use the pointer part
                if self.use_pointer==True:
                    sys.stdout.write('Pointer is enabled!\r')
                    # Calculate the p_gen
                    W_p_gen=tf.get_variable('W_p_gen', [self.decoding_dim*2+self.encoding_dim*2, 1], dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000))
                    b_p_gen=tf.get_variable('b_p_gen', [1,], dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000))
                    p_gen=tf.matmul(decode_representation, W_p_gen)+b_p_gen                                                              # of shape [self.batch_size, 1]
                    p_gen=tf.tile(p_gen, multiples=[1, self.vocab_size])                                                                 # of shape [self.batch_size, self.vocab_size]

                    # Calculate the pointer_prediction
                    batch_indices=tf.expand_dims(tf.range(0, limit=self.batch_size), 1)             # of shape [self.batch_size, 1]
                    batch_indices=tf.tile(batch_indices, multiples=[1, self.max_encoding_step])     # of shape [self.batch_size, self.max_encoding_step]
                    indices=tf.stack((batch_indices, self.encode_input_batch), axis=2)              # of shape [self.batch_size, self.max_encoding_step, 2]

                    self.pointer_prediction=tf.scatter_nd(indices=indices, updates=self.encode_att, shape=[self.batch_size, self.vocab_size])       # of shape [self.batch_size, self.vocab_size]
                    self.prediction=tf.multiply(self.prediction, p_gen)+tf.multiply(self.pointer_prediction, 1-p_gen)                               # of shape [self.batch_size, self.vocab_size]

                else:
                    sys.stdout.write('Pointer is disabled!\r')

                self.prediction=tf.nn.softmax(self.prediction, dim=-1)                        # of shape [self.batch_size, self.vocab_size]
                prediction_sequence.append(tf.reshape(self.prediction, [self.batch_size, 1, self.vocab_size]))

            self.encode_att_final=self.encode_att_history[-1]

        # Calculate the output probability, decoded token and the loss value of the model
        self.predictions=tf.concat(prediction_sequence, axis=1)         # of shape [self.batch_size, self.max_decoding_step, self.vocab_size]
        def add_epsilon(dist, epsilon=sys.float_info.epsilon):
            epsilon_mask = tf.ones_like(dist) * epsilon
            return epsilon_mask + dist
        self.predictions=add_epsilon(self.predictions)

        self.decode_wordidx=tf.argmax(self.predictions, axis=2)         # of shape [self.batch_size, self.max_decoding_step]

        # For decoder
        top_value, self.top_idx=tf.nn.top_k(self.predictions, self.batch_size)
        self.top_value=tf.log(top_value)                                # log likelihood
        print('\n PGNet Model Constructed!!')

    def build_classifier(self):
        if self.loss_type in ['max_likelihood',]:
            self.embedded_ground_truth=tf.one_hot(indices=self.decode_ground_truth, depth=self.vocab_size, axis=-1, dtype=tf.float32)   # of shape [self.batch_size, self.max_decoding_step, self.vocab_size]
            self.padded_loss=tf.log(tf.reduce_sum(tf.multiply(self.embedded_ground_truth, self.predictions), axis=2))         # of shape [self.batch_size, self.max_decoding_step]

            self.loss=-tf.reduce_sum(tf.multiply(self.padded_loss, tf.cast(self.decode_mask, tf.float32)))/tf.reduce_sum(tf.cast(self.decode_mask, tf.float32))         # TODO: can be modified: how to calculate the 'average' loss
        elif self.loss_type in ['sent2vec',]:
            if not 'sent2vec' in self.sub_network or self.sub_network['sent2vec']==None:
                raise Exception('sent2vec sub-network is not successfully loaded')

            # Ordinary part
            self.embedded_ground_truth=tf.one_hot(indices=self.decode_ground_truth, depth=self.vocab_size, axis=-1, dtype=tf.float32)   # of shape [self.batch_size, self.max_decoding_step, self.vocab_size]
            self.padded_loss=tf.log(tf.reduce_sum(tf.multiply(self.embedded_ground_truth, self.predictions), axis=2))         # of shape [self.batch_size, self.max_decoding_step]

            self.loss=-tf.reduce_sum(tf.multiply(self.padded_loss, tf.cast(self.decode_mask, tf.float32)))/tf.reduce_sum(tf.cast(self.decode_mask, tf.float32))         # TODO: can be modified: how to calculate the 'average' loss

            # Sentence Embedding part
            stacked_embedding_matrix=tf.tile(tf.reshape(self.embeddings, [1, self.vocab_size, self.embedding_dim]),
                multiples=[self.batch_size, 1, 1])
            summary_input_embedding=tf.matmul(self.predictions, stacked_embedding_matrix)
            document_input_embedding=self.encode_embedding_output

            summary_vector=self.sub_network['sent2vec'].link_forward(summary_input_embedding)       # of shape [self.batch_size, feature_vector_dim]
            document_vector=self.sub_network['sent2vec'].link_forward(document_input_embedding)     # of shape [self.batch_size, feature_vector_dim]

            dot_product=tf.reduce_sum(tf.multiply(summary_vector, document_vector), axis=1) # of shape [self.batch_size]
            summary_vector_norm=tf.norm(summary_vector, ord=2, axis=1)          # of shape [self.batch_size]
            document_vector_norm=tf.norm(document_vector, ord=2, axis=1)        # of shape [self.batch_size]

            self.cosine_list=tf.div(dot_product, tf.multiply(summary_vector_norm, document_vector_norm))
            self.cosine_loss=tf.reduce_sum(1-self.cosine_list)

            self.loss=tf.multiply(1-self.sent2vec_lambda, self.loss)+tf.multiply(self.sent2vec_lambda, self.cosine_loss)
        else:
            raise ValueError('Unrecognized Loss Type: %s'%self.loss_type)

        print('Optimizer\'s name = %s'%self.update_policy['name'])
        if self.update_policy['name'].lower() in ['sgd', 'stochastic gradient descent']:
            learning_rate=self.update_policy['learning_rate']
            if not 'momentum' in self.update_policy or self.update_policy['momentum']<0.01:
                print('momentum is disabled!')
                self.optimizer=tf.train.GradientDescentOptimizer(learning_rate)
            else:
                print('momentum is enabled!')
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

        # Apply gradient clip
        print('gradient clip is applied, max = %.2f'%self.grad_clip_norm)
        gradients=self.optimizer.compute_gradients(self.loss)
        clipped_gradients=[]
        for grad,var in gradients:
            if grad==None:
                clipped_gradients.append((grad,var))
            else:
                clipped_gradients.append((tf.clip_by_value(grad,-self.grad_clip_norm,self.grad_clip_norm),var))
        self.update=self.optimizer.apply_gradients(clipped_gradients)

    def project(self, variables, output_size, bias, scope_name):
        '''
        >>> Project a vector or list of vectors into a lower or higher dimenaional space.
            >>> variables, tensor or a list of tensors of shape [batch_size, vector_dim]
            >>> output_size, float, the output dimension
            >>> bias, if None, no bias term, else the starting point of bias, an numpy array of shape [output_size,] or a scalar, meaning the same value for the whose vector
            >>> scope_name, the name of variable scope
        '''

        if variables in [None, [], ()]:
            raise ValueError('Invalid variable to be projected %s'%str(variables))
        if not isinstance(variables, (tuple, list)):
            variables=[variables,]
        if isinstance(bias, (int, float)):
            bias=np.ones([output_size],dtype=np.float32)*bias

        total_length=0
        shape_list=[var.get_shape().as_list() for var in variables]
        for shape in shape_list:
            if len(shape)!=2:
                raise ValueError('The dimension of the tensor must be 2, tensors of shape %s detected'%str(shape))
            else:
                total_length+=shape[1]

        with tf.variable_scope(scope_name) as scope:
            input_matrix=variables[0] if len(variables)==1 else tf.concat(variables, axis=1)
            W=tf.get_variable('W', [total_length, output_size], dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10000))
            if not isinstance(bias,(type(None))):
                b=tf.get_variable('b', dtype=tf.float32, initializer=tf.constant(bias))
                return tf.matmul(input_matrix, W)+b
            else:
                return tf.matmul(input_matrix, W)

    def run_encoder(self, encode_input_batch_v, encode_input_length_v):
        '''
        >>> For beam search decoding, run the encoder part only

        >>> encode_input_batch_v: encoder's input batch, of shape [self.batch_size, self.max_encoding_step]
        >>> encode_input_length_v: length of encoder's input sequence, of shape [self.batch_size]
        >>> if the input shape does not match the batch_size and is not 1, raise Exception, if it is 1, do broadcasting

        Value to return
        >>> encoding_states_v: encoder's all hidden states, 
        >>> init_decode_state_v: initial value value of the decode state
        '''
        assert encode_input_batch_v.shape[0]==encode_input_length_v.shape[0], \
            'The number of instances in encode_input_batch and encode_input_length do not match, %s v.s. %s'%(
                str(encode_input_batch_v.shape), str(encode_input_length_v.shape))

        if not encode_input_batch_v.shape[0] in [1, self.batch_size]:
            raise ValueError('The number of instances inside a batch should either be 1 or batch_size')

        # Broadcast
        if encode_input_batch_v.shape[0]==1:
            encode_input_batch_v=[encode_input_batch_v for _ in xrange(self.batch_size)]
            encode_input_length_v=[encode_input_length_v for _ in xrange(self.batch_size)]
            encode_input_batch_v=np.array(encode_input_batch_v).reshape([self.batch_size, -1])
            encode_input_length_v=np.array(encode_input_length_v).reshape([self.batch_size,])

        feed_dict={self.encode_input_batch:encode_input_batch_v, self.encode_input_length:encode_input_length_v}
        if self.sess==None:
            raise Exception('Model can not run the encoding process before a Session is launched!')
        encoding_states_v, init_decode_state_v=self.sess.run([self.encoding_states, self.init_decode_state], feed_dict=feed_dict)
        return encoding_states_v, init_decode_state_v

    def decode_one_step(self, encode_input_batch_v, init_decode_state_v, lastest_token, encoding_states_v, decoding_states_v, encode_att_init_v):
        '''
        >>> For beam search decoding, run the decoder for one time.

        >>> lastest_token: tensor of shape [self.batch_size, 1]
        >>> encoding_states_v: list of LSTMStateTuple, states of the encoder, of shape [self.batch_size, self.max_encoding_step, 2*self.encoding_dim]
        >>> decoding_sfates_v: list of LSTMStateTuple, states of the previous decoder states, of shape [self.batch_size, #Decoded States, self.decoding_dim]
        >>> encode_att_history_v: tensor of shape [self.batch_size, 1], intra-attention on encoding states

        Value to return:
        >>> h_decode_states_v: LSTMStateTuple (c,h)
        >>> h_decode_states_v: the projected decode states that will later be used to calculate intra-decoder-attention
        >>> encode_att_final_v: the lastest updated intra-encoder-attention history
        >>> prediction_v: the predicted distribution of the following word, of shape [self.batch_size, self.vocab_size]
        '''
        assert self.max_decoding_step==1, 'max_decoding_step must be 1, while now it is %d'%self.max_decoding_step
        feed_dict={self.init_decode_state:init_decode_state_v, self.decode_input_batch:lastest_token, self.encoding_states:encoding_states_v,
            self.init_decode_hidden_states:decoding_states_v, self.encode_att_init:encode_att_init_v, self.encode_input_batch: encode_input_batch_v}
        if self.sess==None:
            raise Exception('Model can not run the decoding process before a Session is launched!')

        decode_state_v, h_decode_states_v, encode_att_final_v, top_value_v, top_idx_v =self.sess.run([self.lstm_decode_state, self.h_decode_states,
            self.encode_att_final, self.top_value, self.top_idx], feed_dict=feed_dict)
        h_decode_states_v=h_decode_states_v.reshape([self.batch_size, 1, self.decoding_dim])
        return decode_state_v, h_decode_states_v, encode_att_final_v, top_value_v, top_idx_v

    def train_validate_test_init(self, gpu_prop=-1, save_ckpt_num=10000, loaded_params=[]):
        '''
        >>> preparation for training, evaluation and testing
        >>> gpu_prop: float, the proportion of GPU memory usage
        >>> save_ckpt_num: int, maximum number of check point saved by the self.saver
        >>> loaded_params: list<tf.tensor>, list of external variables
        '''
        if self.sess==None:
            print('Initializing the session for the model')
            if gpu_prop>0 and gpu_prop<1:
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop)
                self.sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
                print('GPU memory usage fraction = %.2f'%gpu_prop)
            else:
                config=tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth=True
                self.sess=tf.Session(config=config)
                print('GPU memory usage mode: flexible and growth allowed')
        else:
            print('The model has already initialized, no need to initilize twice.')
        self.sess.run(tf.global_variables_initializer())
        print('Variables are initialized')
        # print(tf.global_variables())
        param_this_model=[var for var in tf.global_variables() if not var in loaded_params]
        # print(param_this_model)
        self.saver=tf.train.Saver(param_this_model, max_to_keep=save_ckpt_num)
        print('Saver initialized, #Maximum saved check point = %d'%save_ckpt_num)
        return

    def train(self, encoding_input, encoding_length, decoding_input, ground_truth, decode_mask):
        assert self.mode=='train', 'train function is called for a non-train model'
        train_dict={self.encode_input_batch:encoding_input, self.encode_input_length:encoding_length,
            self.decode_input_batch:decoding_input, self.decode_ground_truth:ground_truth, self.decode_mask:decode_mask,
            self.init_decode_hidden_states:np.array([], dtype=np.float32).reshape([self.batch_size, 0, self.decoding_dim])}
        _, loss, predictions, decode_wordidx=self.sess.run([self.update, self.loss, self.predictions, self.decode_wordidx], feed_dict=train_dict)
        return loss, predictions, decode_wordidx


    def validate(self, encoding_input, encoding_length, decoding_input, ground_truth, decode_mask):
        assert self.mode in ['eval', 'train'], 'validate function is called for a non-eval or non-train model'
        validate_dict={self.encode_input_batch:encoding_input, self.encode_input_length:encoding_length,
            self.decode_input_batch:decoding_input, self.decode_ground_truth:ground_truth, self.decode_mask:decode_mask,
            self.init_decode_hidden_states:np.array([], dtype=np.float32).reshape([self.batch_size, 0, self.decoding_dim])}
        loss, predictions, decode_wordidx=self.sess.run([self.loss, self.predictions, self.decode_wordidx], feed_dict=validate_dict)
        return loss, predictions, decode_wordidx

    def dump_params(self,file2dump):
        '''
        >>> Save the parameters
        >>> file2dump: str, file to store the parameters
        '''
        saved_path=self.saver.save(self.sess, file2dump)
        print('parameters are saved in file %s'%saved_path)

    def load_params(self,file2load):
        '''
        >>> Load the parameters
        '''
        self.saver.restore(self.sess, file2load)
        print('parameters are imported from file %s'%file2load)


    def train_validate_test_end(self,):
        print('model %s has finished training, validation and test'%self.name)
        self.sess.close()
        self.sess=None
        self.saver=None


