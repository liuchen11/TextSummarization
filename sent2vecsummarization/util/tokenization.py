import os
import re
import sys

from py2py3 import *
# from nltk import TweetTokenizer
# from nltk.tokenize import StanfordTokenizer

def tokenize(tknzr, sentence, to_lower=True):
    '''
    >>> tknzr: a tokenizer implementing the NLTK tokenizer interface
    >>> sentence: a string to be tokenized
    >>> to_lower: lowercasting or not
    '''
    sentence=sentence.strip()
    if type(sentence)==str:
        sentence=sentence.decode('utf8',errors='ignore').encode('utf8',errors='ignore')
    sentence=' '.join([format_token(x) for x in tknzr.tokenize(sentence)])
    if to_lower:
        sentence=sentence.lower()
    
    sentence=re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',sentence) #replace urls by <url>
    sentence=re.sub('(\@[^\s]+)','<user>',sentence) #replace @user268 by <user>
    filter(lambda word: ' ' not in word, sentence)
    return sentence

def format_token(token):
    replacement_dict={'-LRB-':'(', '-RRB-':')', '-LSB-':'[', '-RSB-':']',
        '-LCB-':'{', '-RCB-':'}'}

    if token in replacement_dict:
        return replacement_dict[token]
    else:
        return token

def tokenize_sentences(tknzr, sentences, to_lower=True):
    '''
    >>> tknzr: a tokenizer implementing the NLTK tokenizer interface
    >>> sentences: a list of sentences
    >>> to_lower: lowercasting or not
    '''
    return [tokenize(tknzr, s, to_lower) for s in sentences]
