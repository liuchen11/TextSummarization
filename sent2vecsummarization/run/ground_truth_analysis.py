import os
import sys
sys.path.insert(0,'./util')
if sys.version_info.major==3:
    xrange=range
    import pickle as cPickle
else:
    import cPickle
import loader
import xml_parser
import numpy as np
import sent2vec_wrapper
from py2py3 import *

def compare_text_summary(data_folders,batch_size,sent2vec_param,output_file,metric='cosine'):
    file_list=[]
    for data_folder in data_folders:
        for file_name in os.listdir(data_folder):
            if os.path.isfile(data_folder+os.sep+file_name) and file_name.split('.')[-1] in ['summary']:
                file_list.append(data_folder+os.sep+file_name)
    print('Detected %d valid documents in total'%len(file_list))

    sent2vec_model=sent2vec_wrapper.Sent2VecWrapper(sent2vec_param)

    document_list=[]
    summary_list=[]
    valid_document_list=[]
    for idx, file_name in enumerate(file_list):
        sys.stdout.write('Loading %d/%d = %.1f%%\r'%(idx+1,len(file_list),float(idx+1)/float(len(file_list))*100))
        sys.stdout.flush()
        parsed_results=loader.parse_document(file_name=file_name)
        if parsed_results==None:
            continue
        sentence_in_document=parsed_results['sentences']
        document=' '.join(sentence_in_document)
        sentence_in_summary=parsed_results['highlights']
        summary=' '.join(sentence_in_summary)
        #TMP
        summary_tokens=document.split(' ')
        length=len(summary.split(' '))
        summary_tokens=summary_tokens*3
        np.random.shuffle(summary_tokens)
	summary_tokens=summary_tokens[:length]
        summary=' '.join(summary_tokens)

        document_list.append(document)
        summary_list.append(summary)
        valid_document_list.append(file_name)

    file_list=valid_document_list
    assert len(file_list)==len(document_list), 'len(file_list)=%d, len(document_list)=%d'%(len(file_list),len(document_list))
    assert len(file_list)==len(summary_list), 'len(file_list)=%d, len(summary_list)=%d'%(len(file_list),len(summary_list))

    document_summary_pair_num=len(file_list)
    batch_num=int((document_summary_pair_num-1)/batch_size)+1
    result_list=[]
    for idx in xrange(batch_num):
        sys.stdout.write('Processing %d/%d batch ...\r'%(idx+1,batch_num))
        sys.stdout.flush()
        document_slice=document_list[idx*batch_size:(idx+1)*batch_size]
        summary_slice=summary_list[idx*batch_size:(idx+1)*batch_size]
        sentence_embeddings=sent2vec_model.get_sentence_embeddings(
            sentence_list=document_slice+summary_slice, ngram=sent2vec_param['ngram'], model=sent2vec_param['model'])
        assert len(sentence_embeddings)==(len(document_slice)+len(summary_slice))
        document_embedding=sentence_embeddings[:len(document_slice)]
        summary_embedding=sentence_embeddings[len(document_slice):]
        for document_vec, summary_vec in zip(document_embedding, summary_embedding):
            if metric in ['cosine']:
                inner_dot=np.dot(document_vec,summary_vec)
                result=inner_dot/np.linalg.norm(document_vec)/np.linalg.norm(summary_vec)
                result_list.append(result)
            elif metric in ['distance']:
                distance_vec=np.array(document_vec)-np.array(summary_vec)
                result=np.linalg.norm(distance_vec)
                result_list.append(result)
            else:
                raise ValueError('Unrecognized metric: %s'%metric)

    assert len(file_list)==len(result_list), 'len(file_list)=%d, len(result_list)=%d'%(len(file_list),len(result_list))
    information=zip(file_list,document_list,summary_list,result_list)
    if sys.version_info.major==2:
        information=sorted(information,lambda x,y:-1 if x[3]<y[3] else 1)
    else:
        information=sorted(information,key=lambda x:x[3], reverse=False)
    to_dump={'summary':{'metric':metric, 'document_num':document_summary_pair_num, 'source':data_folders,
    'average':np.mean(result_list), 'median':np.median(result_list)}, 'details':information}
    cPickle.dump(to_dump, open(output_file,'wb'))

if __name__=='__main__':

    if len(sys.argv)!=2:
        print('Usage: python ground_truth_analysis.py <config_file>')
        exit(0)

    hyper_params=xml_parser.parse(sys.argv[1],flat=False)
    name=hyper_params['name']
    local_params=hyper_params[name]
    if name in ['compare_text_summary',]:
        data_folders=local_params['data_folders']
        batch_size=local_params['batch_size']
        sent2vec_param=local_params['sent2vec_param']
        output_file=local_params['output_file']
        metric='cosine' if not 'metric' in local_params else local_params['metric']
        compare_text_summary(data_folders=data_folders,batch_size=batch_size,sent2vec_param=sent2vec_param,
            output_file=output_file,metric=metric)
    else:
        raise ValueError('Unrecognized function name: %s'%name)

