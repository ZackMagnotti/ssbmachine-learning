'''
Author : Zack Magnotti
Email : zack@magnotti.net
Date : 3/19/2021

Python module containing generator functions
to feed the "clips" dataset (created with clippify.py)
into the training loops for SSBML-Base-Model
and SSBML-Transfer-Model.
'''

import numpy as np
import pickle
import random
import math
from tensorflow import one_hot
from src.util import characters, id_from_char, char_from_id
import os
from os.path import join, splitext

def valid_files(file_list):
    '''Given a list of filenames, return a list of only .pkl filenames'''
    return [f for f in file_list if splitext(f)[1] == '.pkl']

def get_batch(batch_filenames, batch_dir):
    '''
    Unpickles and returns the contents of the files
    listed in batch_filenames, which are located in 
    the directory batch_dir

    Parameters
    -----------
    batch_filenames (list) : filenames of datapoints for this batch
    batch_dir (string) : path to directory to fetch batch from
    
    Outputs (yield)
    -----------
    batch (list) : datapoints (dictionaries) for this batch
    '''
    batch_abspaths = [join(batch_dir, f) for f in batch_filenames]
    batch = [pickle.load(open(abspath, 'rb')) for abspath in batch_abspaths]
    return batch

def character_data(
        input_directory, 
        batch_size = 32,
        num_batches = None,
        repeat = False,
        onehot = True,
        shuffle = True
    ):
    ''' 
    Fetches data from given directory in batches

    Parameters
    -----------
    input_directory (string) : directory from which to fetch data
    batch_size (int) : number of documents per batch
    repeat (bool) : if true, generator loops back after exhausting data, if false, generator stops when data runs out
    onehot (bool) : whether or not to return labels in onehot form
    shuffle (bool) : whether or not to shuffle data 
    
    Outputs (yield)
    -----------
    batch_istreams (ndarray)
    batch_labels (array | ndarray)
    '''

    while True:

        filenames = valid_files(os.listdir(input_directory))

        if shuffle:
            random.shuffle(filenames)
        
        for i in range(0, len(filenames), batch_size):

            # if num_batches specified
            # stop generator once that many batches have been yielded
            if num_batches and i//batch_size >= num_batches :
                return 

            # get batch data
            try:
                batch_filenames = filenames[i:i+batch_size]
            except IndexError:
                batch_filenames = filenames[i:]
            finally:
                batch = get_batch(batch_filenames, input_directory)

            # extract istreams and labels
            batch_istreams = [clip['istream'].toarray() for clip in batch]
            batch_labels = [id_from_char[clip['character']] for clip in batch]

            # convert istreams to single ndarray
            batch_istreams = np.stack(batch_istreams, axis=0)

            if onehot:
                batch_labels = one_hot(batch_labels, 26)

            yield batch_istreams, batch_labels
        
        else:
            if not repeat:
                return 

def player_data(
        player_dir,
        anonymous_dir,
        batch_size = 32,
        repeat = False,
        shuffle = True,
        ratio = 1,
        onehot = False,
    ):
    ''' 
    Fetches data from given directories, and yields a blend of both
    datasets specified by the ratio provided (default is 1, for 50/50 split) 

    Parameters
    -----------
    player_dir (string) : directory containing the player's data
    anonymous_dir (string) : directory containing random data that is not the player's
    batch_size (int) : number of documents per batch
    repeat (bool | int) : if true generator loops back after exhausting data.
                          if false generator stops when data runs out.
                          if integer, loop the given number of times
    onehot (bool) : whether or not to return labels in onehot form
    shuffle (bool) : whether or not to shuffle data 
    ratio (int | float) : ratio of Anonymous games against given player's games (Anonymous / Player) 
    
    Outputs (yield)
    -----------
    batch_istreams (ndarray)
    batch_labels (array | ndarray)
    '''
    
    if repeat is False:
        repeat = 1
    if repeat is True:
        repeat = np.inf

    if repeat < 0:
        raise ValueError
    if not ratio > 0:
        raise ValueError

    player_filenames = valid_files(os.listdir(player_dir))
    anonymous_filenames = valid_files(os.listdir(anonymous_dir))

    # represents uninitialized data
    player_current_index = 0
    anonymous_current_index = 0

    while True:
        player_batch_size = np.random.binomial(
            n = batch_size, 
            p = 1 / (ratio + 1)
        )
        anonymous_batch_size = batch_size - player_batch_size

        # =====================
        #   get player batch
        # =====================

        # indexing
        pstart = player_current_index
        pend = pstart + player_batch_size
        player_current_index += player_batch_size

        try:
            player_batch = get_batch(player_filenames[pstart:pend], player_dir)
        # If data is exhausted and repeat counter is not at 0, 
        # decrement repeat counter, reset current index, and shuffle clips
        except IndexError:
            if not repeat > 0:
                return
            repeat -= 1
            player_current_index = 0
            random.shuffle(player_filenames)
            player_batch = get_batch(player_filenames[pstart:pend], player_dir)

        # list of tuple(istream, label)
        player_batch_tuples = [(clip['istream'].toarray(), 1.0) for clip in player_batch]

        # ======================
        #  get anonymous batch
        # ======================

        # indexing
        npstart = anonymous_current_index
        npend = npstart + anonymous_batch_size
        anonymous_current_index += anonymous_batch_size

        try:
            anonymous_batch = get_batch(anonymous_filenames[npstart:npend], anonymous_dir)
        except IndexError:
            anonymous_current_index = 0
            random.shuffle(anonymous_filenames)
            anonymous_batch = get_batch(anonymous_filenames[npstart:npend], anonymous_dir)

        # list of tuple(istream, label)
        anonymous_batch_tuples = [(clip['istream'].toarray(), 0.0) for clip in anonymous_batch]

        # ==============
        #  mix batches 
        # ==============

        # Mix batchs, separate istreams from labels, and convert both into ndarrays
        batch_tuples = player_batch_tuples + anonymous_batch_tuples
        random.shuffle(batch_tuples)
        batch_istreams = np.stack([istream for istream, _ in batch_tuples], axis=0)
        batch_labels = np.array([label for _, label in batch_tuples])

        if onehot:
            batch_labels = one_hot(batch_labels, 2)

        yield batch_istreams, batch_labels
