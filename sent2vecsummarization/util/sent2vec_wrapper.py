import os
import sys
import time
import numpy as np

from subprocess import call
from nltk import TweetTokenizer
from nltk.tokenize import StanfordTokenizer

import xml_parser
import tokenization
from py2py3 import *

class Sent2VecWrapper(object):

    def __init__(self, hyper_params):

        self.sent2vec_root=hyper_params['sent2vec_root']
        self.snlp_root=hyper_params['snlp_root']
        self.pretrain_model_root=hyper_params['pretrain_model_root']

        self.fasttext_path=os.path.join(self.sent2vec_root,'fasttext')
        self.snlp_jar=os.path.join(self.snlp_root,'stanford-postagger.jar')
        self.unigram_wiki_path=os.path.join(self.pretrain_model_root,'sent2vec_wiki_unigrams')
        self.bigram_wiki_path=os.path.join(self.pretrain_model_root,'sent2vec_wiki_bigrams')
        self.unigram_tweet_path=os.path.join(self.pretrain_model_root,'sent2vec_tweet_unigrams')
        self.bigram_tweet_path=os.path.join(self.pretrain_model_root,'sent2vec_tweet_bigrams')

        assert os.path.exists(self.snlp_jar), 'Stanford nlp library is missing.'
        assert os.path.exists(self.unigram_wiki_path), 'Pretrained model based on unigram wikipedia is missing.'

    def _read_embeddings(self, embedding_path):

        with open(embedding_path, 'r') as fopen:
            embeddings=[]
            for line in fopen:
                line=''.join(['[', line.replace(' ',','), ']'])
                embeddings.append(eval(line))
            return embeddings

    def _dump_text(self, file_path, X, Y=None):
        '''
        >>> file_path: where to dump the information
        >>> X: list of sentences
        >>> Y: list of labels, if applicable
        '''
        with open(file_path, 'w') as fopen:
            if Y is not None:
                for x,y in zip(X,Y):
                    fopen.write('__label__'+str(y)+' '+x+' \n')
            else:
                for x in X:
                    fopen.write(x+' \n')

    def _preprocessed_sentence_embedding(self, sentence_list, model_path, fasttext_path):
        '''
        >>> sentence_list: list of sentences
        >>> model_path: a path to sent2vec *.bin file
        >>> fasttext_path: a path to fasttext executable
        '''
        timestamp=str(time.time())
        test_path=os.path.abspath(self.sent2vec_root+os.sep+timestamp+'_fasttext_test.txt')
        embedding_path=os.path.abspath(self.sent2vec_root+os.sep+timestamp+'_fasttext_embedding.txt')
        self._dump_text(file_path=test_path,X=sentence_list)
#        print('model_path=%s, test_path=%s, embedding_path=%s'%(model_path,test_path,embedding_path))
        call('%s print-sentence-vectors %s < %s > %s'%(fasttext_path,model_path,test_path,embedding_path),shell=True)
        embeddings=self._read_embeddings(embedding_path)
        os.remove(test_path)
        os.remove(embedding_path)
        assert(len(sentence_list)==len(embeddings)), 'len(sentence_list)=%d, len(embeddings)=%d'%(len(sentence_list), len(embeddings))
        return np.array(embeddings)

    def get_sentence_embeddings(self, sentence_list, ngram, model):
        '''
        >>> sentence_list: list of sentences
        >>> ngram: 'unigram' or 'bigram'
        >>> model: 'wiki','twitter' or 'concat_wiki_twitter'
        '''
        wiki_embeddings=None
        twitter_embeddings=None

        if model in ['wiki', 'concat_wiki_twitter']:
            tkzr=StanfordTokenizer(self.snlp_jar, encoding='utf-8')
            s=' <delimiter> '.join(sentence_list)
            s=s.decode('utf-8', errors='ignore')
            tokenized_sentence_SNLP_ori=tokenization.tokenize_sentences(tkzr, [s])
            tokenized_sentence_SNLP=tokenized_sentence_SNLP_ori[0].split(' <delimiter> ')
            try:
                assert(len(tokenized_sentence_SNLP)==len(sentence_list))
            except:
                print('Stanford tokenizer does not work, use plain preprocessor instead')
                tokenized_sentence_SNLP=map(lambda x:x.lower().decode('utf-8', errors='ignore'),sentence_list)
                assert(len(tokenized_sentence_SNLP)==len(sentence_list))
            if ngram=='unigram':
                wiki_embeddings=self._preprocessed_sentence_embedding(tokenized_sentence_SNLP,
                    self.unigram_wiki_path, self.fasttext_path)
            elif ngram=='bigram':
                wiki_embeddings=self._preprocessed_sentence_embedding(tokenized_sentence_SNLP,
                    self.bigram_wiki_path, self.fasttext_path)
            else:
                raise ValueError('Unrecognized N-grams: %s'%ngram)
        if model in ['twitter', 'concat_wiki_twitter']:
            tkzr=TweetTokenizer()
            tokenized_sentence_TWEET=tokenization.tokenize_sentences(tkzr, sentence_list)
            if ngram=='unigram':
                twitter_embeddings=self._preprocessed_sentence_embedding(tokenized_sentence_TWEET,
                    self.unigram_tweet_path, self.fasttext_path)
            elif ngram=='bigram':
                twitter_embeddings=self._preprocessed_sentence_embedding(tokenized_sentence_TWEET,
                    self.bigram_tweet_path, self.fasttext_path)
            else:
                raise ValueError('Unrecognized N-grams: %s'%ngram)

        if model in ['wiki']:
            return wiki_embeddings
        elif model in ['twitter']:
            return twitter_embeddings
        elif model in ['concat_wiki_twitter']:
            return np.concatenate([wiki_embeddings, twitter_embeddings], axis=1)
        else:
            raise ValueError('Unrecognized model: %s'%model)


