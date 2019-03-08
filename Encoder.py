#!/usr/bin/python
# In[2]:

import argparse
import pandas as pd
from pipe import *
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, repeat, islice
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Bidirectional, RepeatVector, Activation, Softmax, TimeDistributed
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import csv

# In[2]:

parser = argparse.ArgumentParser()
parser.add_argument("excel_path", help="path to the excel spreadsheet")
parser.add_argument("field_name", help="what is the name of the field?")
parser.add_argument("code_name", help="short codename for field i.e. 'sku'")
parser.add_argument("-b", "--batch_size", default=1000, type=int, help="how big do you want the batch to be")
parser.add_argument("-s", "--seq_length", default=10, type=int, help="how long do you want the sequence length to be")
parser.add_argument("-n", "--num_epochs", default=10, type=int, help="How many epochs to run for")
parser.add_argument("-m", "--latent_dim", default=10, type=int, help="dimensionality of the latent dimension?")

args = parser.parse_args()

# In[2]:
# for interactive running
path = './9_TonerYield.xlsx'
name = 'HW SKU Description'
batch_size = 100
seq_length =  10
num_epochs = 1
latent_dim = 5
run_shortname =  "sku"



# In[2]:
path = args.excel_path #'data/4_SuppliesShipment (filtered).xlsx'
name = args.field_name  #'Toner Part No. (SKU)'
batch_size = args.batch_size
seq_length =  args.seq_length
num_epochs = args.num_epochs
latent_dim = args.latent_dim
run_shortname =  args.code_name

# In[2]:
print(f"loading {path}")
sheet = pd.read_excel(path, index_col=0)

# In[5]:
overall_count = list( sheet[name].values )  \
        | as_list \
        | count

uniq_count = list( sheet[name].values ) \
        | select( lambda x: str(x).lower() ) \
        | sort( key=lambda x: x, reverse=True )  \
        | uniq \
        | count

print( f"Unique Count: {uniq_count} Overall Count: {overall_count}" )

# In[5]:

codes = ( list( sheet[name].values )    
        | select( lambda x: str(x).lower() )
        | groupby( lambda x: x )     
        | select( lambda g: (g[0], g[1] | count) )    
        | sort( key=lambda x: x[1], reverse=True )     
        | as_list )

print( codes | take(40) | as_list )

# In[6]:


how_many_inplot = 30
skus = codes | select(lambda c: c[0]) | take(how_many_inplot) | as_list
counts  = codes | select(lambda c: c[1]) | take(how_many_inplot)  | as_list

h = plt.bar(skus, counts)
h = plt.xticks(rotation='vertical')
fig = plt.gcf()
fig.set_size_inches(18,10)
h = plt.xlabel(name)
h = plt.ylabel('Count')
h = plt.title('%s (%d unique)' % ( name, len( codes ) ) )

plt.savefig(f'{run_shortname}.pdf')
# clear the figure so we don't get the next one superimposed
plt.clf()

# How many unique tokens do we have in this signal?

# In[7]:

# create token maps

tokens = list(sheet[name].astype('str'))     \
        | select( lambda e: list(str(e).lower()) )     \
        | traverse     \
        | dedup     \
        | as_list

token_map = zip( range( len(tokens) ), tokens )     | as_dict
token_map_inv = {v: k for k, v in token_map.items()}

print(token_map)

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None): 
   return islice(pad_infinite(iterable, padding), size)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def is_float(value):
  try:
    float(value)
    return True
  except:
    return False

def GetBatch(df, token_map, batchsize=5, cycle=True):
    
    cycle_mode = True

    while cycle_mode:

        if(not cycle): cycle_mode = False

        c = chunker(range(overall_count), batchsize)
        
        for chunk in c:

            X = list( df[list(chunk)] )                 \
                | select( lambda l: list(str(str(l).lower()))                     \
                | select( lambda t: token_map_inv[t] ) | as_list )                 \
                | select( lambda l: pad(l, size=seq_length, padding=0 ) | as_list )                 \
                | as_list
            
            X = to_categorical( np.array( X ), num_classes=len(token_map))

            F = list( df[list(chunk)] ) \
                | select( lambda l: list(l) \
                    | select(lambda y: float(y)/10 if is_float(y) else 0) \
                    | as_list ) \
                | select( lambda l: pad(l, size=seq_length, padding=0 ) | as_list ) \
                | as_list 

            F = np.array( F )
            
            # let's be a bit clever here and add an additional float for tokens \in {0,1,..9}
            # so it can learn
            # it's an ordinal albeit in a sequence, it should then be able to learn
            # to count
            X_bigger = np.zeros((len(chunk),seq_length,len(token_map)+1))
            X_bigger[:,:,0:len(token_map)] = X
            X_bigger[:,:,len(token_map)] = F

            # yield same thing twice because the label is the same as the signal
            yield (X_bigger,X_bigger)

# In[9]:


num_encoder_tokens = len(token_map)

model = Sequential()
model.add(Bidirectional(LSTM(latent_dim, activation='relu'), input_shape=(seq_length, num_encoder_tokens+1)))
model.add(RepeatVector(seq_length))
model.add(Bidirectional(LSTM(latent_dim, activation='relu', return_sequences=True)))
model.add(TimeDistributed(Dense(latent_dim*2)))
model.add(TimeDistributed(Dense(latent_dim)))
model.add(TimeDistributed(Dense(num_encoder_tokens+1)))

data_gen = GetBatch(
    sheet[name], 
    token_map, 
    batchsize=batch_size)

opt = Adam(lr=0.01)

# Run training
model.compile(opt, 
              loss='mse')

callbacks = [ EarlyStopping(
        monitor='loss', 
        min_delta=0, 
        patience=10, 
        verbose=0, 
        mode='auto',  
        baseline=None, 
        # this will restore the best weights back to the model
        restore_best_weights=True) ]

history = model.fit_generator(data_gen,
          epochs=num_epochs, 
          steps_per_epoch=int(overall_count/batch_size),
          callbacks=callbacks)

# In[70]:


loss = history.history['loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(f'{run_shortname}_convergence.png')
# In[32]:

encoder = Model(inputs=model.inputs, outputs=model.layers[0].output)

# save the encoder so that we can use it again later on new data
model.save(f'{run_shortname}_encoder.h5')

all_data = GetBatch(
    sheet[name], 
    token_map, 
    batchsize=batch_size,
    cycle=False)

all_embeddings = np.zeros( ( overall_count, latent_dim*2 )  )

print('one-hot encode all data and predict in batches')

for i, x in enumerate( all_data ):
    
    fr = (i*batch_size)
    to = ([(i*batch_size)+batch_size, overall_count] | min)

    print(f'batch {i} ({fr}=>{to})')

    all_embeddings[ fr:to ] = encoder.predict(x[0])

print('saving embeddings')
np.savetxt(f"{run_shortname}_embeddings.csv", all_embeddings, delimiter="\t")

# Also save out the SKUs so we can visualise with the tensorflow embedding visualiser

# In[68]:


print('saving labels')

labels_all = list( sheet[name].values )   \
| select( lambda x: (str(x).lower() ) )     \
| as_list



print(f'Labels count: {labels_all | count}')

with open(f"{run_shortname}_labels.csv",'w', encoding="utf-8") as resultFile:
    
    for item in labels_all:
        nobreak = item.replace('\n', '').replace('\r', '')

        resultFile.write(f"{nobreak}\n")