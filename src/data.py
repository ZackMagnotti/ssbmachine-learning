import numpy as np
import pickle
import random
import math
from tensorflow import one_hot
from src.util import characters, id_from_char, char_from_id
import os
from os.path import join, splitext

def valid_files(file_list):
    return [f for f in file_list if splitext(f)[1] == '.pkl']

def get_batch(batch_filenames, input_directory):
    batch_abspaths = [join(input_directory, f) for f in batch_filenames]
    batch = [pickle.load(open(abspath, 'rb')) for abspath in batch_abspaths]
    return batch

def character_data(input_directory, 
                   batch_size = 32,
                   num_batches = None,
                   repeat = False,
                   onehot = True,
                   shuffle = True):
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

def player_data(player_dir,
                nonplayer_dir,
                batch_size = 32,
                num_batches = None,
                shuffle = True,
                ratio = 1):
    ''' 
    Fetches data from given directories, and yields a blend of both
    datasets specified by the ratio provided (default is 1, for 50/50 split) 

    Parameters
    -----------
    player_dir (string) : directory containing the player's data
    nonplayer_dir (string) : directory containing random data that is not the player's
    batch_size (int) : number of documents per batch
    repeat (bool) : if true, generator loops back after exhausting data, if false, generator stops when data runs out
    onehot (bool) : whether or not to return labels in onehot form
    shuffle (bool) : whether or not to shuffle data 
    ratio (int | float) : ratio of player data to non-player data (player/non-player) 
    
    Outputs (yield)
    -----------
    batch_istreams (ndarray)
    batch_labels (array | ndarray)
    '''
    player_filenames = valid_files(os.listdir(player_dir))
    player_batch_size = int(ratio * batch_size // (ratio + 1))
    player_current_index = math.inf

    nonplayer_filenames = valid_files(os.listdir(nonplayer_dir))
    nonplayer_batch_size = batch_size - player_batch_size
    nonplayer_current_index = math.inf

    while True:
        
        # =====================
        #   get player batch
        # =====================

        # if player data is exhausted or uninitialized
        if player_current_index + player_batch_size >= len(player_filenames):
            player_current_index = 0
            random.shuffle(player_filenames)
        
        pstart = player_current_index
        pend = pstart + player_batch_size
        player_batch = get_batch(player_filenames[pstart:pend], input_directory)

        # list of tuple(istream, label)
        # label for player is 1
        player_batch_tuples = [(clip['istream'].toarray(), 1) for clip in player_batch]
        
        # ======================
        #  get nonplayer batch
        # ======================

        # if nonplayer data is exhausted or uninitialized
        if nonplayer_current_index + nonplayer_batch_size >= len(nonplayer_filenames):
            nonplayer_current_index = 0
            random.shuffle(nonplayer_filenames)
        
        npstart = nonplayer_current_index
        npend = npstart + nonplayer_batch_size
        nonplayer_batch = get_batch(nonplayer_filenames[npstart:npend], input_directory)

        # list of tuple(istream, label)
        # label for nonplayer is 0
        nonplayer_batch_tuples = [(clip['istream'].toarray(), 0) for clip in nonplayer_batch]
        
        # ==============
        #  mix batches 
        # ==============

        batch_tuples = player_batch_tuples + nonplayer_batch_tuples
        random.shuffle(batch_tuples)

        # get istreams and labels from tuples
        batch_istreams = [istream for istream, _ in batch_tuples]
        batch_labels = [label for _, label in batch_tuples]

        # convert istreams to single ndarray
        batch_istreams = np.stack(batch_istreams, axis=0)

        batch_labels = np.array(batch_labels)

        yield batch_istreams, batch_labels
