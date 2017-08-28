import os
import sys
sys.path.insert(0,'./util')
sys.path.insert(0,'./model')
import numpy as np

import xml_parser

class SentSimTemplate(object):

    def __init__(self, hyper_param):

        self.name=hyper_param['name']

    def compare(self, sentence1, sentence2):
        raise NotImplementedError('This is the basic class, where the compare function is not implemented yet')

class SentSim_Sent2Vec(object):

    def __init__(self, hyper_param):

        super.__init__(hyper_param)



