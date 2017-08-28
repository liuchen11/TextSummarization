import os
import re
import sys
import time
import numpy as np
if sys.version_info.major==2:
    import cPickle
else:
    import pickle as cPickle
    xrange=range
from nltk.tokenize import StanfordTokenizer

sys.path.insert(0, './model')
sys.path.insert(0, './util')

import loader
from py2py3 import *
import sent2vec_distributor

def tokenize(tknzr, sentence, to_lower=True):
    """Arguments:
        - tknzr: a tokenizer implementing the NLTK tokenizer interface
        - sentence: a string to be tokenized
        - to_lower: lowercasing or not
    """
    sentence = sentence.strip()
    sentence = ' '.join([format_token(x) for x in tknzr.tokenize(sentence)])
    if to_lower:
        sentence = sentence.lower()
    sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',sentence) #replace urls by <url>
    sentence = re.sub('(\@[^\s]+)','<user>',sentence) #replace @user268 by <user>
    filter(lambda word: ' ' not in word, sentence)
    return sentence

def format_token(token):
    """"""
    if token == '-LRB-':
        token = '('
    elif token == '-RRB-':
        token = ')'
    elif token == '-RSB-':
        token = ']'
    elif token == '-LSB-':
        token = '['
    elif token == '-LCB-':
        token = '{'
    elif token == '-RCB-':
        token = '}'
    return token

def tokenize_sentences(tknzr, sentences, to_lower=True):
    """Arguments:
        - tknzr: a tokenizer implementing the NLTK tokenizer interface
        - sentences: a list of sentences
        - to_lower: lowercasing or not
    """
    return [tokenize(tknzr, s, to_lower) for s in sentences]

if len(sys.argv)<6:
    print('Usage: python analysis_SE_similarity.py <tokenizer> <fasttext_path> <fasttext_model> <data_folder> <saved_pkl>')
    exit(0)

tokenizer_file=sys.argv[1]
fasttext_path=sys.argv[2]
fasttext_model=sys.argv[3]
data_folder=sys.argv[4]
saved_pkl=sys.argv[5]

# Collect useful file:
file_list=[]
for file_name in os.listdir(data_folder):
    if file_name.split('.')[-1] in ['summary',]:
        file_list.append(data_folder+os.sep+file_name)

print('There are %d files detected in the folder %s'%(len(file_list), data_folder))

document_info_list=[]
for idx, file_name in enumerate(file_list):
    sys.stdout.write('Loading files %d/%d=%.1f%%\r'%(idx+1, len(file_list), float(idx+1)/float(len(file_list))*100))
    sys.stdout.flush()
    document_info=loader.parse_document(file_name=file_name, replace_entity=True)
    if document_info!=None:
        document=' '.join(document_info['sentences'])
        summary=' '.join(document_info['highlights'])
        document=document.replace('\n','')
        summary=summary.replace('\n','')
        document_info_list.append((file_name, document, summary))
print('Information already loaded')

# Construction of the sentence embedding distributor
distributor=sent2vec_distributor.EmbeddingDistributor(fasttext_path=fasttext_path, fasttext_model=fasttext_model)
# Construct the tokenizer
tokenizer=StanfordTokenizer(tokenizer_file, encoding='ascii')

result_list=[]
for idx, (file_name, document, summary) in enumerate(document_info_list):
    sys.stdout.write('Loading file %s %d/%d=%.1f%%\r'%(file_name, idx+1, len(document_info_list), float(idx+1)/float(len(document_info_list))*100))
    sys.stdout.flush()

    # Remove non-ascii tokens
    document=document.decode('ascii', errors='replace').encode('ascii', errors='replace')
    summary=summary.decode('ascii', errors='replace').encode('ascii', errors='replace')

    # Remove some useless tags
    document=document.replace('***', '')
    summary=summary.replace('***', '')

    # Cut the summary within 100 words
    summary=' '.join(summary.split(' ')[:100])

    summary_tokens=summary.split(' ')
    summary_tokens_num=len(summary_tokens)
    sentence_list=[document,]
    for token_idx in xrange(summary_tokens_num):
        summary_prefix=' '.join(summary_tokens[:token_idx+1])
        sentence_list.append(summary_prefix)

    # s='<split>'.join(sentence_list)
    # s=tokenize_sentences(tokenizer, [s])
    # sentence_list=s[0].split('<split>')
    sent_embeddings=distributor.get_tokenized_sents_embeddings(sents=sentence_list)
    # time.sleep(0.1)

    document_embedding=sent_embeddings[0]
    cosine_value_list=[]
    distance_list=[]
    document_embedding_norm=np.linalg.norm(document_embedding, 2)

    for token_idx in xrange(summary_tokens_num):
        prefix_embedding=sent_embeddings[token_idx+1]
        prefix_embedding_norm=np.linalg.norm(prefix_embedding, 2)
        cosine_value=np.dot(document_embedding, prefix_embedding)/document_embedding_norm/prefix_embedding_norm if prefix_embedding_norm>1e-8 else 0.0
        distance=np.linalg.norm(document_embedding-prefix_embedding, 2)
        cosine_value_list.append(cosine_value)
        distance_list.append(distance)
    result_list.append({'file_name':file_name, 'document':document, 'summary':summary, 'cosine_value_list':cosine_value_list, 'distance_list':distance_list})

cPickle.dump(result_list, open(saved_pkl, 'wb'))
print('Everything is completed and the information is saved in %s'%saved_pkl)

