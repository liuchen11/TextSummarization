import sys
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
sys.path.insert(0,'util')
from py2py3 import *

import numpy as np

def rouge(reference, candidate, N):
    '''
    >>> reference: list<str>, list of reference sentences
    >>> candidate: list<str>, list of candidate sentences
    >>> N: int, rouge-N
    '''
    # remove \n
    reference=map(lambda x: x.lower() if x[-1]!='\n' else x[:-1].lower(), reference)
    candidate=map(lambda x: x.lower() if x[-1]!='\n' else x[:-1].lower(), candidate)

    # Extract N-grams of both lists
    reference_ngram=[]
    candidate_ngram=[]

    for sentence in reference:
        tokens=sentence.split(' ')
        if len(tokens)<N:
            continue
        for token_idx in xrange(len(tokens)-N+1):
            reference_ngram.append(tokens[token_idx:token_idx+N])

    for sentence in candidate:
        tokens=sentence.split(' ')
        if len(tokens)<N:
            continue
        for token_idx in xrange(len(tokens)-N+1):
            candidate_ngram.append(tokens[token_idx:token_idx+N])

    reference_ngram_num=len(reference_ngram)
    candidate_ngram_num=len(candidate_ngram)

    hits=map(lambda ngram: 1 if ngram in candidate_ngram else 0, reference_ngram)

    hits_num=np.sum(hits)
    total_ngram=len(hits)

    rouge_score=float(hits_num)/float(total_ngram)

    return rouge_score, reference_ngram_num, candidate_ngram_num

def rouge_l(reference, candidate):
    '''
    >>> reference: list<str>, list of reference sentences
    >>> candidate: list<str>, list of candidate sentences
    >>> return the longest pattern match
    '''
    # remove \n
    reference=map(lambda x: x.lower() if x[-1]!='\n' else x[:-1].lower(), reference)
    candidate=map(lambda x: x.lower() if x[-1]!='\n' else x[:-1].lower(), candidate)

    # clean
    reference_token_list=('\n'.join(reference)).split(' ')
    candidate=('\n'.join(candidate)).split('\n')

    # KMP algorithm to find the longest pattern match
    longest_match=0
    for sentence in candidate:
        tokens=sentence.split(' ')
        index_list=[]

        pt=0
        for idx,token in enumerate(tokens):
            index_list.append(pt)
            if token==tokens[pt] and idx!=pt:
                pt+=1
            else:
                pt=0

        pt=0
        for idx,token in enumerate(reference_token_list):
            if pt==len(tokens):
                break
            elif token==tokens[pt]:
                pt+=1
            else:
                if pt>longest_match:
                    longest_match=pt
                while True:
                    pt=index_list[pt]
                    if token==token[pt]:
                        pt+=1
                        break
                    if pt==0:
                        break
        if pt>longest_match:
            longest_match=pt

    return longest_match

if __name__=='__main__':

    reference=['a b a d','a b a b c a b']
    candidate=['a b a c','b a b c a']

    rouge_1_score,_,_=rouge(reference, candidate, N=1)
    rouge_2_score,_,_=rouge(reference, candidate, N=2)
    rouge_l_score=rouge_l(reference, candidate)

    print('R-1 score: %.2f'%rouge_1_score)
    print('R-2 score: %.2f'%rouge_2_score)
    print('R-l score: %.2f'%rouge_l_score)

