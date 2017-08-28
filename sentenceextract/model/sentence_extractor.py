import sys
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
sys.path.insert(0,'./util')
from py2py3 import *
import os
import numpy as np
import tensorflow as tf

sys.path.insert(0,'./model')
from __init__ import *

class sentence_extractor(base_net):

    def __init__(self, hyper_params):
        '''
        >>> Construct a CNN-LSTM model
        >>> hyper_params: dict, a dictionary containing all hyper parameters
            >>> batch_size: int, batch size
            >>> sequence_length: int, maximum sentence length
            >>> sequence_num: int, maximum number of sentences in a document
            >>> rnn_size: int, the number of neurons in LSTM
            >>> mlp_neurons: list<int>, the number of neurons in each layer of mlp
            >>> class_num: int, number of categories
            >>> vocab_size: int, vocabulary size
            >>> embedding_dim: int, dimension of word embeddings
            >>> filter_sizes: list<int>, different kinds of filter size i.e window size
            >>> feature_map: int, number of feature maps for different filters
            >>> update_policy: dict, update policy
            >>> embedding_matrix: optional, numpy.array, initial embedding matrix of size [self.vocab_size, self.embedding_dim]
        '''
        self.batch_size=hyper_params['batch_size']
        self.sequence_length=hyper_params['sequence_length']
        self.sequence_num=hyper_params['sequence_num']
        self.rnn_size=hyper_params['rnn_size']
        self.mlp_neurons=hyper_params['mlp_neurons']
        self.class_num=hyper_params['class_num']
        self.vocab_size=hyper_params['vocab_size']
        self.embedding_dim=hyper_params['embedding_dim']
        self.filter_sizes=hyper_params['filter_sizes']
        self.feature_map=hyper_params['feature_map']
        self.separator_trainable=hyper_params['separator_trainable'] if 'separator_trainable' in hyper_params else True
        self.update_policy=hyper_params['update_policy']
        self.grad_clip_norm=hyper_params['grad_clip_norm'] if 'grad_clip_norm' in hyper_params else 1.0
        self.name='sentence extraction model' if not 'name' in hyper_params else hyper_params['name']

        # input extension
        self.input_extend_types=hyper_params['input_extend_types'] if 'input_extend_types' in hyper_params else None
        self.input_extensions={}
        self.word_representation_dim=self.embedding_dim

        self.sess=None

        if not 'embedding_matrix' in hyper_params:
            print('Word embeddings are initialized from scrach')
            self.embedding_matrix=tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_dim],-1.0,1.0),dtype=tf.float32)
        else:
            print('Pre-trained word embeddings are imported')
            assert(hyper_params['embedding_matrix'].shape[0]==self.vocab_size)
            assert(hyper_params['embedding_matrix'].shape[1]==self.embedding_dim)
            self.embedding_matrix=tf.Variable(hyper_params['embedding_matrix'],dtype=tf.float32)

        self.inputs=tf.placeholder(tf.int32,shape=[self.batch_size, self.sequence_num,self.sequence_length]) # None represents the number of sequences
        self.masks=tf.placeholder(tf.int32,shape=[self.batch_size, self.sequence_num])                       # Masks - 1 if the sentence is valid else 0
        self.labels=tf.placeholder(tf.int32,shape=[self.batch_size, self.sequence_num])                      # Labels of each sentences
        self.ratio=tf.placeholder(tf.float32,shape=())                                                       # Ratio of P_t using ground truth or prediction

        self.embedding_output=tf.nn.embedding_lookup(self.embedding_matrix,self.inputs)                      # shape = [self.batch_size, sequence_num, self.sequence_length, self.embedding_dim]

        if self.input_extend_types!=None:
            for input_extend_type in self.input_extend_types:
                if input_extend_type.lower() in ['entity_bit']:
                    if not 'entity_bit' in self.input_extensions:
                        self.input_extensions['entity_bit']=tf.placeholder(tf.float32,shape=[self.batch_size,self.sequence_num,self.sequence_length,1])
                    self.embedding_output=tf.concat([self.embedding_output,self.input_extensions['entity_bit']],axis=3)
                    self.word_representation_dim+=1
                else:
                    raise ValueError('Unrecognized input extension type: %s'%input_extend_type)

        input_slices=tf.split(self.embedding_output,self.sequence_num,1)

        # Construct sentence_embedding for each sentences in the documents
        sentence_embeddings=[]
        with tf.variable_scope('CNN') as scope:
            for slice_idx in xrange(self.sequence_num):
                if slice_idx!=0:
                    scope.reuse_variables()
                parts=[]
                for filter_idx, filter_size in enumerate(self.filter_sizes):
                    convpool_output=self.convpool(input_data_3d=input_slices[slice_idx],
                        filter_width=filter_size, name='filter%d'%filter_idx)       # of size [self.batch_size, self.feature_map]
                    parts.append(convpool_output)
                current_embedding=parts[0]
                for part in parts[1:]:
                    current_embedding=tf.add(part,current_embedding)
                sentence_embeddings.append(current_embedding)

        # LSTM-based document encoding part
        with tf.variable_scope('LSTM') as scope:
            LSTM_cell=tf.contrib.rnn.BasicLSTMCell(self.rnn_size)        # Contruct a LSTM cell
            state_sequences=[]
            state=LSTM_cell.zero_state(self.batch_size,tf.float32)
            for sentence_idx in xrange(self.sequence_num):
                if sentence_idx!=0:
                    scope.reuse_variables()
                _, state=LSTM_cell(sentence_embeddings[sentence_idx],state)
                state_sequences.append(state[1])                    # record the hidden state in each step

        # Sentence Extraction part
        predictions=[]
        expand_matrix=tf.ones([1,self.feature_map],dtype=tf.float32)

        with tf.variable_scope('Buffer_input') as scope:
            buffer_input=tf.Variable(tf.zeros([1, self.feature_map], dtype=tf.float32),trainable=self.separator_trainable,dtype=tf.float32)
        buffer_input=tf.matmul(tf.ones([self.batch_size, 1],dtype=tf.float32), buffer_input)
       
        with tf.variable_scope('LSTM',reuse=True) as scope:
            _, state=LSTM_cell(buffer_input,state)
        with tf.variable_scope('MLP',reuse=False) as scope:
            mlp_input=tf.concat([state_sequences[0],state[1]],axis=1)
            sentence_prediction=self.mlp(input_data=mlp_input, hidden_sizes=self.mlp_neurons)
            predictions.append(sentence_prediction)
        for sentence_idx in xrange(self.sequence_num-1):
            input_sentence_embedding=tf.multiply(tf.matmul(predictions[-1]*self.ratio+tf.to_float(self.labels[:,sentence_idx:sentence_idx+1])*(1.0-self.ratio)
                ,expand_matrix),sentence_embeddings[sentence_idx])
            with tf.variable_scope('LSTM',reuse=True) as scope:
                _, state=LSTM_cell(input_sentence_embedding,state)
            with tf.variable_scope('MLP',reuse=True) as scope:
                mlp_input=tf.concat([state_sequences[sentence_idx+1],state[1]],axis=1)
                sentence_prediction=self.mlp(input_data=mlp_input, hidden_sizes=self.mlp_neurons)
                predictions.append(sentence_prediction)

        # Generate final prediction and loss
        self.final_prediction=tf.concat(predictions, axis=1)        # of size [self.batch_size, self.sequence_num]
        self.loss= -tf.reduce_mean(tf.multiply(tf.to_float(self.masks), tf.multiply(
            tf.to_float(self.labels), tf.log(self.final_prediction))+tf.multiply(1.0 - tf.to_float(self.labels), tf.log(1.0-self.final_prediction))))

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
        elif self.update_policy['name'].lower() in ['none']:
            print('Warning: Optimizer is disabled!! Not possible to train the model')
            self.optimizer=None
        else:
            raise ValueError('Unrecognized Optimizer Category: %s'%self.update_policy['name'])

        # Apply gradient clip
        if self.optimizer!=None:
            print('gradient clip is applied, max = %.2f'%self.grad_clip_norm)
            gradients=self.optimizer.compute_gradients(self.loss)
            clipped_gradients=[(tf.clip_by_value(grad,-self.grad_clip_norm,self.grad_clip_norm),var) for grad,var in gradients]
            self.update=self.optimizer.apply_gradients(clipped_gradients)
        else:
            self.update=None

    def convpool(self, input_data_3d, filter_width, name, stddev=0.02):
        '''
        >>> Construct a convolutional-pooling layer
        >>> input_data: tf.Variable, input data of size [self.batch_size, self.sequence_length, self.word_representation_dim]
        >>> filter_width: int, the width of the filter
        >>> name: str, the name of this layer
        >>> stddev: float, the standard derivation of the weight for initialization
        '''
        input_data=tf.expand_dims(tf.reshape(input_data_3d,shape=[self.batch_size,self.sequence_length,self.word_representation_dim]),-1)
        with tf.variable_scope(name):
            W=tf.get_variable(name='W',shape=[filter_width,self.word_representation_dim,1,self.feature_map],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv_output=tf.nn.conv2d(input_data,W,strides=[1,1,1,1],padding='VALID')
            pool_output=tf.nn.max_pool(conv_output,ksize=[1,self.sequence_length-filter_width+1,1,1],
                strides=[1,self.sequence_length-filter_width+1,1,1],padding='VALID')
            return tf.reshape(pool_output,shape=[self.batch_size,self.feature_map])

    def mlp(self, input_data, hidden_sizes, name='mlp', stddev=0.02):
        '''
        >>> Construct a multilayer perceptron model
        >>> input_data: tf.Variable, input data of size [self.batch_size, rnn_size*2]
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
            input_dim=hidden_sizes[-1]
            output_dim=1
            W=tf.get_variable(name='W_final',shape=[input_dim,output_dim],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
            b=tf.get_variable(name='b_final',shape=[output_dim,],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
            data=tf.nn.sigmoid(tf.add(tf.matmul(data,W),b))                     # of size [self.batch_size, 1]
            return data

    def train_validate_test_init(self,gpu_memory_fraction=1.0/4.0):
        '''
        >>> Initialize the training validation and test phrase
        '''
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        self.sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        print('gpu memory fraction = %.2f'%gpu_memory_fraction)
        init=tf.global_variables_initializer()
        self.sess.run(init)
        self.decision_func=np.vectorize(lambda x: 1 if x>0.5 else 0)

    def train(self,inputs,masks,labels,ratio,extend_part):
        '''
        >>> Training process on a batch data
        >>> inputs: np.array, of size [self.batch_size, self.sequence_num,self.sequence_length]
        >>> masks: np.array, of size [self.batch_size, self.sequence_num]
        >>> labels: np.array, of size [self.batch_size, self.sequence_num]
        >>> ratio: float, 1 means totally using the prediction, 0 means totally using the ground truth
        >>> extend_part: map<str -> tensors>, additional information that maps the extend_tag to tensor needed
        '''
        if self.update==None:
            raise Exception('This model does not have a valid optimizer while the train function is called')

        train_dict={self.inputs:inputs, self.masks:masks, self.labels:labels, self.ratio:ratio}
        for key in self.input_extensions:
            if not key in extend_part:
                raise ValueError('%s is necessary as the extended part of input of neural network, while it is missing'%(key))
            else:
                train_dict[self.input_extensions[key]]=extend_part[key]

        self.sess.run(self.update,feed_dict=train_dict)
        final_prediction_this_batch, loss_this_batch=self.sess.run([self.final_prediction, self.loss],feed_dict=train_dict)
        final_prediction_this_batch=self.decision_func(final_prediction_this_batch)
        return final_prediction_this_batch, loss_this_batch

    def validate(self,inputs,masks,labels,ratio,extend_part):
        '''
        >>> Validation phrase
        >>> Parameter table is the same as self.train
        '''
        validate_dict={self.inputs:inputs, self.masks:masks, self.labels:labels, self.ratio:ratio}
        for key in self.input_extensions:
            if not key in extend_part:
                raise ValueError('%s is necessary as the extended part of input of neural network, while it is missing'%(key))
            else:
                validate_dict[self.input_extensions[key]]=extend_part[key]

        final_prediction_this_batch, loss_this_batch=self.sess.run([self.final_prediction, self.loss],feed_dict=validate_dict)
        final_prediction_this_batch=self.decision_func(final_prediction_this_batch)
        return final_prediction_this_batch, loss_this_batch

    def test(self,inputs,masks,extend_part,fine_tune=False):
        '''
        >>> Test phrase
        >>> Parameter table is almost the same as self.train
        >>> No labels are provided
        '''
        labels=np.zeros([self.batch_size, self.sequence_num], dtype=np.float32)         # Fake labels
        ratio=1.0                                                                       # ratio has to be 1.0 to make prediction label independent
        test_dict={self.inputs:inputs, self.masks:masks, self.labels:labels, self.ratio:ratio}

        for key in self.input_extensions:
            if not key in extend_part:
                raise ValueError('%s is necessary as the extended part of input of neural network, while it is missing'%(key))
            else:
                test_dict[self.input_extensions[key]]=extend_part[key]

        final_prediction_this_batch,=self.sess.run([self.final_prediction,],feed_dict=test_dict)
        if not fine_tune:
            final_prediction_this_batch=self.decision_func(final_prediction_this_batch)
        return final_prediction_this_batch

    def do_summarization(self, file_list, folder2store, data_generator, n_top=5):
        '''
        >>> Give the top sentences for each of several documents
        >>> file_list: list<str>, list of document files
        >>> folder2store: str, the directory to store the temporary *.info file
        >>> data_generator: util.data_generator, manager the data batch generation
        '''
        document_num_remained=len(file_list)
        raw_text_list=[]
        idx_text_list=[]
        if not os.path.exists(folder2store):
            os.makedirs(folder2store)
        for file in file_list:
            pure_file=file.split(os.sep)[-1]
            dest_file='.'.join(pure_file.split('.')[:-1])+'.info'
            raw_text_list.append(file)
            idx_text_list.append(folder2store+os.sep+dest_file)
        data_generator.gen_idx_from_text(raw_text_list=raw_text_list,idx_text_list=idx_text_list)
        data_generator.init_batch_gen(set_label='tmp',file_list=idx_text_list,permutation=False,force=True)
        stop=False

        file_idx=0
        top_sentence_list=[]
        while not stop:
            input_matrix,masks,labels,stop,extension_part=data_generator.batch_gen(
                set_label='tmp',batch_size=self.batch_size,label_policy='min',extend_tags=self.input_extend_types)
            final_prediction_this_batch=self.test(inputs=input_matrix,masks=masks,extend_part=extension_part,fine_tune=True)
            valid_document=min(self.batch_size,document_num_remained)
            document_num_remained-=valid_document
            final_prediction_this_batch=final_prediction_this_batch[:valid_document]
            for idx,prediction in enumerate(final_prediction_this_batch):
                lines=open(file_list[file_idx],'r').readlines()
                lines=map(lambda x: x if x[-1]!='\n' else x[:-1],lines)
                sentence_num=len(lines)
                real_prediction=prediction[:sentence_num]
                if sys.version_info.major==2:
                    sorted_list=sorted(zip(np.arange(1,sentence_num+1),real_prediction,lines),
                        lambda x,y: 1 if x[1]<y[1] else -1)
                else:
                    sorted_list=sorted(zip(np.arange(1,sentence_num+1),real_prediction,lines),
                        key=lambda x:x[1],reverse=True)
                top_sentence_list.append(sorted_list[:n_top])
                file_idx+=1
        assert(len(top_sentence_list)==len(file_list))
        return top_sentence_list

    def dump_params(self,file2dump):
        '''
        >>> Save the parameters
        >>> file2dump: str, file to store the parameters
        '''
        saver=tf.train.Saver()
        saved_path=saver.save(self.sess, file2dump)
        print('parameters are saved in file %s'%saved_path)

    def load_params(self,file2load,loaded_params=[]):
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
