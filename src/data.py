import numpy as np
import pickle
from tensorflow import one_hot
from src.util import characters, id_from_char, char_from_id
import os
from os.path import join



def data_generator ( input_directory, 
                     batch_size=25,
                     num_batches=None,
                     repeat=False,
                     onehot=True,
                     shuffle=False ) :
    ''' 
    Fetches data from given directory in batches

    Parameters
    -----------
    input_directory (string) : directory from which to fetch data
    batch_size (int) : number of documents per batch
    repeat (bool) : if true, generator loops back after exhausting data, if false, generator stops when data runs out
    query (dict) : optional query to filter what data is fetched
    onehot (bool) : whether or not to return labels in onehot form
    shuffle (bool) : whether or not to shuffle data 
    
    Outputs (yield)
    -----------
    batch_istreams (ndarray)
    batch_labels (array | ndarray)
    '''
    filenames = os.listdir(input_directory)
    
    if shuffle:
        import random.shuffle as shuff
        shuff(filenames)
        
    filenames_static_copy = filenames
    
    while True:
        for i in range(0, len(filenames), batch_size):

            # if num_batches specified
            # stop generator once that many batches have been yielded
            if num_batches and num_batches <= i//batch_size:
                return 

            # get batch data
            try:
                batch_filenames = filenames[i:i+batch_size]
            except IndexError:
                batch_filenames = filenames[i:]

            batch_abspaths = [join(input_directory, f) for f in batch_filenames]
            batch = [pickle.load(open(abspath, 'rb')) for abspath in batch_abspaths]

            # extract istreams and labels
            batch_istreams = [clip['istream'].toarray() for clip in batch]
            batch_labels = [id_from_char[clip['character']] for clip in batch]

            # convert istreams to single array
            batch_istreams = np.stack(batch_istreams, axis=0)

            if onehot:
                batch_labels = one_hot(batch_labels, 26)

            yield batch_istreams, batch_labels

        else:
            if not repeat:
                break
                