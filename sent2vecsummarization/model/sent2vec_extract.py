import os
import sys
sys.path.insert(0,'./util')
import numpy as np

from py2py3 import *
import sent2vec_wrapper

class Sent2VecExtract(object):

    def __init__(self, hyper_params):

        sent2vec_wrapper_params=hyper_params['sent2vec_wrapper']
        self.name=hyper_params['name']
        self.sent2vec=sent2vec_wrapper.Sent2VecWrapper(sent2vec_wrapper_params)
        self.ngram='bigram' if 'ngram' not in hyper_params else hyper_params['ngram']
        self.model='wiki' if 'model' not in hyper_params else hyper_params['model']
        self.n_top=5 if 'n_top' not in hyper_params else hyper_params['n_top']

    def extract(self, document_list, return_idx=False, sort_by_order=True, saved_marks=True):
        sentence_list=[]
        sentence_by_documents=[]
        sentence_num_per_documents=[]

        for document in document_list:
            sentence_this_document=document.split('\n')
            sentence_list=sentence_list+sentence_this_document
            sentence_num_per_documents.append(len(sentence_this_document))
            sentence_by_documents.append(sentence_this_document)

        print('Detected %d documents with %d sentences in total'%(len(document_list),len(sentence_list)))

        assert(len(sentence_by_documents)==len(document_list))
        assert(len(sentence_num_per_documents)==len(document_list))

        document_sentence_list=document_list+sentence_list

        document_sentence_embeddings=self.sent2vec.get_sentence_embeddings(
            sentence_list=document_sentence_list, ngram=self.ngram, model=self.model)
        assert(len(document_sentence_embeddings)==len(document_list)+len(sentence_list))

        document_embeddings=document_sentence_embeddings[:len(document_list)]
        sentence_embeddings=document_sentence_embeddings[len(document_list):]
        assert(len(document_embeddings)==len(document_list))
        assert(len(sentence_embeddings)==len(sentence_list))

        candidate_pre_idx=0
        candidate_post_idx=sentence_num_per_documents[0]
        key_sentences=[]
        for idx, (document_vec,sentence_candidate) in enumerate(zip(document_embeddings, sentence_by_documents)):
            sentence_vec_candidate=sentence_embeddings[candidate_pre_idx:candidate_post_idx]
            assert(len(sentence_vec_candidate)==sentence_num_per_documents[idx])
            assert(len(sentence_candidate)==len(sentence_vec_candidate))
            
            result_list=[]
            for sen_idx, sentence_vec in enumerate(sentence_vec_candidate):
                cosine_value=np.dot(sentence_vec,document_vec)/np.linalg.norm(sentence_vec,2)/np.linalg.norm(document_vec,2)
                result_list.append((sen_idx,cosine_value))

            if sys.version_info.major==2:
                result_list=sorted(result_list, lambda x,y: 1 if x[1]<y[1] else -1)
            else:
                result_list=sorted(result_list, key=lambda x:x[1], reverse=True)

            if idx+1<len(sentence_num_per_documents):
                candidate_pre_idx=candidate_post_idx
                candidate_post_idx+=sentence_num_per_documents[idx+1]

            if return_idx==True:
                if idx==0:
                    print('parameter "return_idx" is set True, parameter "n_top", "sort_by_order" and "saved_marks" is ignored')
                key_sentences.append(result_list)
                continue
            sentence_selected=[]
            top_list=result_list[:self.n_top]
            if sort_by_order==True:
                if sys.version_info.major==2:
                    top_list=sorted(top_list, lambda x,y: -1 if x[0]<y[0] else 1)
                else:
                    top_list=sorted(top_list, key=lambda x: x[0], reverse=False)

            for sen_idx, cosine_value in top_list:
                sentence_selected.append(sentence_candidate[sen_idx] if not saved_marks \
                    else (sentence_candidate[sen_idx], cosine_value, sen_idx))
            key_sentences.append(sentence_selected)

        return key_sentences

