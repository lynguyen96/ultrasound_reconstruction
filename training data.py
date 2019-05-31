
# coding: utf-8

# In[1]:


cd D:\Ly\Python_Beamform\training


# In[2]:


import os
from lip.model import SDA
import datasets.utils as dutils
from itertools import product
import time
start = time. time()

# Load test set
test_set = dutils.load_ius2017_test_set()

end = time. time()
total_running_time=(end - start)


# In[3]:


train_set_sz=len(test_set)*1024


# In[4]:


import numpy as np
train_set = np.zeros((train_set_sz,128), dtype=float)
ind_start = 0


# In[5]:


for seq_ref in test_set:
    data_ref = seq_ref.data[0]
    data=np.transpose(data_ref)
    ind_end = ind_start + data.shape[0]
    train_set[ind_start:ind_end] = data
    ind_start = ind_end


# In[ ]:


# Generate a cross valid set
full_train_set=train_set
valid_set, train_set = dutils.generate_cross_valid_sets(full_set=full_train_set,
                                                        valid_size=2000,
                                                        seed=123456789)


# In[ ]:


# Benchmark launch settings
batch_size = 4096
learning_rate = 0.001
num_epochs = 20
dump_percent = 10
data_dim = train_set.shape[1]
base_dir = os.path.join('networks', 'ius2017')
cp_list = range(50, 99, 5)
lmm_list = [False, True]

for lmm, cp in product(lmm_list, cp_list):
    # SDA model
    model = SDA(data_dim=data_dim, compression_percent=cp, learn_mm=lmm, base_dir=base_dir)

    # Train model
    model.train(learning_rate=learning_rate,
                train_set=train_set,
                valid_set=valid_set,
                num_epochs=num_epochs,
                batch_size=batch_size,
                dump_percent=dump_percent)

