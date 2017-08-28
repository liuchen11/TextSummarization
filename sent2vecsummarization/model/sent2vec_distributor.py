import numpy as np
import pty
import os
import subprocess
import cPickle

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class EmbeddingDistributor(object):

    def __init__(self, fasttext_path, fasttext_model):
        master, slave = pty.openpty()
        
        self._proc = subprocess.Popen(fasttext_path+' print-sentence-vectors '+fasttext_model, shell=True, stdin=subprocess.PIPE, stdout=slave, bufsize=1)
        self._stdin_handle = self._proc.stdin
        self._stdout_handle = os.fdopen(master)

    def get_tokenized_sents_embeddings(self, sents):
        # Deal with big lists
        if len(sents)>20:
            segments=int((len(sents)-1)/20)+1
            sub_embeddings=[]
            for seg_idx in xrange(segments):
                embeddings=self.get_tokenized_sents_embeddings(sents[seg_idx*20:(seg_idx+1)*20])
                sub_embeddings.append(embeddings)
            return np.concatenate(sub_embeddings, axis=0).reshape(len(sents),-1)

        for sent in sents:
            if '\n' in sent:
                raise RuntimeError('New line is not allowed inside a sentence')

        sentence_lines = '\n'.join(sents)+'\n'
        self._stdin_handle.write(sentence_lines.encode())
        self._stdin_handle.flush()
        all_embeddings = []
        for _ in sents:
            res = self._stdout_handle.readline()[:-2] #remove last space and jumpline
            all_embeddings.append(eval('[' + res.replace(' ', ',') + ']'))
        return np.array(all_embeddings)