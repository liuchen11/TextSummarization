import sys
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
sys.path.insert(0,'./util')
from py2py3 import *
import os
import numpy as np
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
sys.path.insert(0,'./util')
from py2py3 import *
import os
import numpy as np
import tensorflow as tf

class convnet(object):

    def __init__(self, hyper_params):
        '''
        >>> Construct a CNN classifier
        >>> hyper_params: a dictionary containing all hyper_params
            >>> batch_size: int, batch size
            >>> sequence_length: int, maximum sentence length
            >>> class_num: int, number of categories
            >>> vocab_size: int, vocabulary size
            >>> embedding_dim: int, dimension of word embeddings
            >>> filter_sizes: list<int>, different kinds of filter size i.e window size
            >>> feature_map: int, number of feature maps for different filters
            >>> hidden states: optional list<int>, hidden states between CNN extractor and decision layer
            >>> update_policy: dict, update policy
            >>> embedding_matrix: optional, numpy.array, initial embedding matrix of size [self.vocab_size, self.embedding_dim]
        '''
        self.batch_size=hyper_params['batch_size']
        self.sequence_length=hyper_params['sequence_length']
        self.class_num=hyper_params['class_num']
        self.vocab_size=hyper_params['vocab_size']
        self.embedding_dim=hyper_params['embedding_dim']
        self.filter_sizes=hyper_params['filter_sizes']
        self.feature_map=hyper_params['feature_map']
        self.hidden_sizes=hyper_params['hidden_sizes'] if 'hidden_sizes' in hyper_params else []
        self.update_policy=hyper_params['update_policy']
        self.grad_clip_norm=hyper_params['grad_clip_norm'] if 'grad_clip_norm' in hyper_params else 1.0
        self.name='convnet' if not 'name' in hyper_params else hyper_params['name']

        # Mark the original variable list
        ori_var_list=tf.global_variables()

        # input extension
        self.input_extend_types=hyper_params['input_extend_types'] if 'input_extend_types' in hyper_params else None
        self.input_extensions={}
        self.word_representation_dim=self.embedding_dim

        self.sess=None

        with tf.variable_scope('convnet') as root_scope:
            self.root_scope=root_scope
            if not 'embedding_matrix' in hyper_params:
                print('Word embeddings are initialized from scratch')
                self.embedding_matrix=tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_dim],-1.0,1.0),dtype=tf.float32)
            else:
                print('Pre-trained word embeedings are imported')
                assert(hyper_params['embedding_matrix'].shape[0]==self.vocab_size)
                assert(hyper_params['embedding_matrix'].shape[1]==self.embedding_dim)
                self.embedding_matrix=tf.Variable(hyper_params['embedding_matrix'],dtype=tf.float32)

            self.inputs=tf.placeholder(tf.int32,shape=[self.batch_size, self.sequence_length])
            self.labels=tf.placeholder(tf.int32,shape=[self.batch_size])

            self.embedding_output=tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)    # of shape [self.batch_size, self.sequence_length, self.embedding_dim]

            if self.input_extend_types!=None:
                for input_extend_type in self.input_extend_types:
                    if input_extend_type.lower() in ['entity_bit']:
                        if not 'entity_bit' in self.input_extensions:
                            self.input_extensions['entity_bit']=tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_length, 1])
                        self.embedding_output=tf.concat([self.embedding_output, self.input_extensions['entity_bit']], axis=2)
                        self.word_representation_dim+=1
                    else:
                        raise ValueError('Unrecognized input extension type: %s'%input_extend_type)

            with tf.variable_scope('CNN') as scope:
                parts=[]
                for filter_idx, filter_size in enumerate(self.filter_sizes):
                    convpool_output=self.convpool(input_data=self.embedding_output,
                        filter_width=filter_size, name='filter%d'%filter_idx)
                    parts.append(convpool_output)
                self.current_embedding=parts[0]
                for part in parts[1:]:
                    self.current_embedding=tf.add(part,self.current_embedding)

            unnormalized_prediction=self.mlp(input_data=self.current_embedding, hidden_sizes=[self.feature_map,]+self.hidden_sizes,
                output_size=self.class_num, name='Classifier')                          # of shape [self.batch_size, self.class_num]
            normalized_prediction=tf.nn.softmax(unnormalized_prediction, dim=-1)        # fix the bug solved by tensorflow 1.1
            embedded_label=tf.nn.embedding_lookup(tf.eye(self.class_num), self.labels)  # of shape [self.batch_size, self.class_num]
            self.prediction=tf.argmax(normalized_prediction, axis=1)
            self.output=normalized_prediction
            self.loss=-tf.reduce_mean(tf.multiply(embedded_label, tf.log(normalized_prediction)))

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

            # apply gradient clip
            print('Gradient clip is applied, max = %.2f'%self.grad_clip_norm)
            gradients=self.optimizer.compute_gradients(self.loss)
            clipped_gradients=[(tf.clip_by_value(grad,-self.grad_clip_norm,self.grad_clip_norm),var) for grad,var in gradients]
            self.update=self.optimizer.apply_gradients(clipped_gradients)

        # Collect the variable list
        post_var_list=tf.global_variables()
        self.var_list=[]
        for var in post_var_list:
            if not var in ori_var_list:
                self.var_list.append(var)

        print('ConvNet Model has been constructed!')

    def convpool(self, input_data, filter_width, name, stddev=0.02):
        '''
        Construct a convolutional network followed by a global max-pooling layer
        >>> input_data: tf.Variable, input data of size [self.batch_size, self.sequence_length, self.word_representation_dim]
        >>> filter_width: int, the width of the filter
        >>> name: str, the name of this layer
        >>> stddev: float, the standard derivation of the weight for initialization
        '''
        input_data=tf.expand_dims(input_data,-1)
        with tf.variable_scope(name):
            W=tf.get_variable(name='W',shape=[filter_width, self.word_representation_dim, 1, self.feature_map],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv_output=tf.nn.conv2d(input_data,W,strides=[1,1,1,1],padding='VALID')
            pool_output=tf.nn.max_pool(conv_output,ksize=[1,self.sequence_length-filter_width+1,1,1],
                strides=[1,self.sequence_length-filter_width+1,1,1],padding='VALID')
            return tf.reshape(pool_output,shape=[self.batch_size,self.feature_map])

    def mlp(self, input_data, hidden_sizes, output_size, name='mlp', stddev=0.02):
        '''
        >>> Construct a multilayer perceptron model
        >>> input_data: tf.Variable, input data of size [self.batch_size, self.feature_map]
        >>> hidden_sizes: list<int>, number of neurons in each hidden layer, including the number of neurons in the input layer
        >>> name: str, the name of this model
        >>> stddev: float, the standard derivation of the weight for initialization
        '''
        with tf.variable_scope(name):
            data=input_data
            for idx,neuron_num in enumerate(hidden_sizes[:-1]):
                input_dim=neuron_num
                output_dim=hidden_sizes[idx+1]
                W=tf.get_variable(name='W%d'%idx,shape=[input_dim,output_dim],
                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                b=tf.get_variable(name='b%d'%idx,shape=[output_dim,],
                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                data=tf.nn.relu(tf.add(tf.matmul(data,W),b))

            if output_size==None:           # Get the feature vectors
                return data

            input_dim=hidden_sizes[-1]
            output_dim=output_size
            W=tf.get_variable(name='W_final',shape=[input_dim,output_dim],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
            b=tf.get_variable(name='b_final',shape=[output_dim,],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
            return tf.add(tf.matmul(data,W),b)      # of size [self.batch_size, output_size]

    def train_validate_test_init(self, gpu_memory_fraction=-1):
        '''
        >>> Initialize the training validation and test phrase
        '''
        if gpu_memory_fraction>0 and gpu_memory_fraction<1:
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            gpu_config=tf.ConfigProto(gpu_options=gpu_options)
            message='GPU memory fraction = %.2f'%gpu_memory_fraction
        else:
            gpu_config=tf.ConfigProto(allow_soft_placement=True)
            gpu_config.gpu_options.allow_growth=True
            message='GPU memory mode: soft increase'

        if self.sess!=None:
            print('Model\'s session cannot be created twice')
        elif tf.get_default_session()!=None:
            print('Directly use the default session and drop configuration parameters')
            self.sess=tf.get_default_session()
        else:
            self.sess=tf.Session(config=gpu_config)
            print(message)
        init=tf.global_variables_initializer()
        self.sess.run(init)

    def train(self, inputs, labels, extend_part):
        '''
        >>> Training process on a batch data
        >>> inputs: np.array, of size [self.batch_size, self.sequence_length]
        >>> labels: np.array, of size [self.batch_size,]
        >>> extend_part: map<str -> tensors>, additional information
        '''
        train_dict={self.inputs:inputs, self.labels:labels}
        for key in self.input_extensions:
            if not key in extend_part:
                raise ValueError('%s is necessary as the extended part of input of neural network, while it is missing'%key)
            else:
                train_dict[self.input_extensions[key]]=extend_part[key]

        _, prediction_this_batch, loss_this_batch=self.sess.run([self.update, self.prediction, self.loss], feed_dict=train_dict)
        return prediction_this_batch, loss_this_batch

    def validate(self, inputs, labels, extend_part):
        '''
        >>> Validation phrase
        >>> Parameter table is same as self.train
        '''
        validate_dict={self.inputs:inputs, self.labels:labels}
        for key in self.input_extensions:
            if not key in extend_part:
                raise ValueError('%s is necessary as the extended part of input of neural network, while it is missing'%key)
            else:
                validate_dict[self.input_extensions[key]]=extend_part[key]

        prediction_this_batch, loss_this_batch=self.sess.run([self.prediction, self.loss], feed_dict=validate_dict)
        return prediction_this_batch, loss_this_batch

    def test(self, inputs, extend_part, fine_tune=False):
        '''
        >>> Test phrase
        >>> Parameter table is same as self.train
        '''
        test_dict={self.inputs:inputs,}
        for key in self.input_extensions:
            if not key in extend_part:
                raise ValueError('%s is necessary as the extended part of input of neural network, while it is missing'%key)
            else:
                test_dict[self.input_extensions[key]]=extend_part[key]

        if fine_tune==False:
            prediction_this_batch,=self.sess.run([self.prediction,], feed_dict=test_dict)
        else:
            prediction_this_batch,=self.sess.run([self.output,], feed_dict=test_dict)

        return prediction_this_batch

    def extract_features(self, inputs, extend_part):
        '''
        >>> Extract features
        >>> Parameter table is same as self.train
        '''
        feature_dict={self.inputs:inputs,}
        for key in self.input_extensions:
            if not key in extended_part:
                raise ValueError('%s is necessary as the extended part of input of neural network, while it is missing'%key)
            else:
                test_dict[self.input_extensions[key]]=extended_part[key]

        features,=self.sess.run([self.current_embedding,], feed_dict=feature_dict)
        return features

    def link_forward(self, input_tensor):
        '''
        >>> input_tensor: tf.Tensor of shape [self.batch_size, None, self.embedding_dim]
        '''

        # Pad or Cut the input_tensor to make sure its shape is [self.batch_size, self.sequence_length, self.embedding_dim]
        input_shape=input_tensor.shape.as_list()
        if input_shape[1]>self.sequence_length:
            input_tensor=input_tensor[:,:self.sequence_length,:]
        elif input_shape[1]<self.sequence_length:
            padding_tensor=tf.nn.embedding_lookup(self.embedding_matrix,tf.fill(dims=[self.batch_size, self.sequence_length-input_shape[1]], value=4))
            input_tensor=tf.concat([input_tensor, padding_tensor], axis=1)

        with tf.variable_scope(self.root_scope, reuse=True):
            assert self.input_extend_types in [None, []], 'input_extend_types have to be None, but now it is %s'%str(self.input_extend_types)

            with tf.variable_scope('CNN') as scope:
                parts=[]
                for filter_idx, filter_size in enumerate(self.filter_sizes):
                    convpool_output=self.convpool(input_data=input_tensor, filter_width=filter_size, name='filter%d'%filter_idx)
                    parts.append(convpool_output)
                current_embedding=parts[0]
                for part in parts[1:]:
                    current_embedding=tf.add(part, current_embedding)

            feature_vector=self.mlp(input_data=current_embedding, hidden_sizes=[self.feature_map,]+self.hidden_sizes,
                output_size=None, name='Classifier')

        return feature_vector

    def do_summarization(self, file_list, folder2store, data_generator, n_top=5):
        '''
        >>> Give the top sentences for each of several documents
        >>> file_list: list<str>, list of document files
        >>> folder2store: str, the directory to store the temporary *.info file
        >>> data_generator: util.data_generator, manager the data batch generation
        >>> n_top: number of top sentences to preserve
        '''
        raise NotImplementedError('Current model is naive_extractor, where "do_summarization" function is not implemented')

    def dump_params(self, file2dump):
        '''
        >>> Save the parameters
        >>> file2dump: str, file to store the parameters
        '''
        saver=tf.train.Saver()
        saved_path=saver.save(self.sess, file2dump)
        print('Parameters are saved in file %s'%saved_path)

    def load_params(self, file2load, loaded_params=[]):
        if loaded_params==None or len(loaded_params)==0:
            param2load=self.var_list
        else:
            param2load=[]
            for var in tf.global_variables():
                if not var in loaded_params:
                    param2load.append(var)
        saver=tf.train.Saver(param2load)
        saver.restore(self.sess, file2load)
        print('Parameter are imported from file %s'%file2load)

    def train_validate_test_end(self):
        '''
        >>>End current training validation and test phrase
        '''
        self.sess.close()
        self.sess=None

