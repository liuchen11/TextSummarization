# This is the base class of network

class base_net(object):

    def __init__(self, hyper_params):
        print('Construting a neural network')

    def do_summarization(self, file_list, folder2store, data_generator, n_top=5):
        '''
        >>> Do NOTHING 
        '''
        print('This is the virtual function of the base class')