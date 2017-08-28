import os
import sys
if sys.version_info.major==2:
    import cPickle
else:
    import pickle as cPickle

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, './')

if len(sys.argv)<2:
    print('Usage: python plot_SE_similarity.py <pkl_file>')
    exit(0)

result_list=cPickle.load(open(sys.argv[1], 'rb'))
print('There are %d documents\' detected!'%(len(result_list)))

max_sequence_length=100

distance_matrix=[]
cosine_value_matrix=[]

reverse=True if max_sequence_length<0 else False
max_sequence_length=max_sequence_length if max_sequence_length>0 else -max_sequence_length

for idx, result in enumerate(result_list):
    sys.stdout.write('Loading %d/%d=%.1f%%\r'%(idx+1, len(result_list),
        float(idx)/float(len(result_list))*100))
    distance_matrix.append(result['distance_list'][:max_sequence_length] if reverse==False \
        else list(reversed(result['distance_list']))[:max_sequence_length])
    cosine_value_matrix.append(result['cosine_value_list'][:max_sequence_length] if reverse==False \
        else list(reversed(result['cosine_value_list']))[:max_sequence_length])

cosine_value_final=[]
distance_value_final=[]

for idx in xrange(max_sequence_length):
    distance_pt=[1 if len(info)>idx else 0 for info in distance_matrix]
    cosine_pt=[1 if len(info)>idx else 0 for info in cosine_value_matrix]
    distance_padded_list=[info[idx] if len(info)>idx else 0. for info in distance_matrix]
    cosine_padded_list=[info[idx] if len(info)>idx else 0. for info in cosine_value_matrix]

    if idx==0:
        cPickle.dump([distance_pt, cosine_pt, distance_padded_list, cosine_padded_list], open('saved.pkl', 'wb'))

    cosine_value_final.append(np.sum(cosine_padded_list)/np.sum(cosine_pt))
    distance_value_final.append(np.sum(distance_padded_list)/np.sum(distance_pt))

plt.title('The cosine and distance value distribution of sentence embeddings, reverse=%s'%str(reverse))
plt.plot(np.arange(1, max_sequence_length+1), cosine_value_final, color='r', label='cosine')
plt.plot(np.arange(1, max_sequence_length+1), distance_value_final, color='g', label='distance')
plt.legend()
plt.show()
