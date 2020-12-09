from pymongo import MongoClient
import numpy as np
import tensorflow as tf
import pickle

from src.util import characters, id_from_char, char_from_id

'''
    TODO: 
        split this up into data_generator, xgenerator, and ygenerator
'''

def get_next_clip(cur, step, repeat, skip, limit):
    '''
        get every step-th clip from cursor

        if step is 2, get every other clip, etc
    '''
    for _ in range(step):
        try:
            clip = next(cur)

        # if cur is empty but
        # but repeat is true
        # reset cursor
        except StopIteration:
            if repeat:
                cur.rewind()
                if skip:  cur.skip(skip)
                if limit: cur.limit(limit)
                clip = next(cur)
            else:
                raise
    
    return clip

def data_generator(clip_collection,
                   batch_size = 100,
                   skip=None,
                   step=1,
                   repeat=False,
                   mode=None,
                   limit=None,
                   onehot=True):
        
    cur = clip_collection.find()
    
    if skip:
        cur.skip(skip)
    
    if limit:
        cur.limit(limit)
        
    while cur.alive:
        
        xi = []
        yi = []
        
        for _ in range(batch_size):
            # get the next clip
            try:
                clip = get_next_clip(cur, step, repeat, skip, limit)
            
            # abort loop and yield what there is so far
            # this will be the last yield
            except StopIteration:
                break

            # if next clip is fetched successfully 
            # append its info to the lists
            else:
                if mode != 'Y' and mode != 'y':
                    xi.append(pickle.loads(clip['istream']).toarray())
                yi.append(id_from_char[clip['character']])

        if xi == [] and yi == []:
            raise StopIteration
        
        if mode != 'Y' and mode != 'y':
            xi = np.stack(xi, axis=0)
        
        if onehot:
            yi = tf.one_hot(yi, 26)
        
        if mode == 'X' or mode == 'x':
            yield xi
            
        elif mode == 'Y' or mode == 'y':
            yield yi
            
        elif mode:
            raise ValueError("mode must be 'X' or 'Y' or None")

        else:
            yield xi, yi

def clip_generator(clip_collection,
                   batch_size = 100,
                   skip=None,
                   step=1,
                   repeat=False,
                   mode=None,
                   limit=None,
                   onehot=True):
        
    cur = clip_collection.find()
    
    if skip:
        cur.skip(skip)
    
    if limit:
        cur.limit(limit)
        
    while cur.alive:
        
        xi = []
        
        for _ in range(batch_size):
            # get the next clip
            try:
                clip = get_next_clip(cur, step, repeat, skip, limit)
            
            # abort loop and yield what there is so far
            # this will be the last yield
            except StopIteration:
                break

            # if next clip is fetched successfully 
            # append its info to the lists
            else:
                xi.append(pickle.loads(clip['istream']).toarray())

        if xi == []:
            raise StopIteration
        
        yield np.stack(xi, axis=0)

def label_generator(clip_collection,
                    batch_size = 100,
                    skip=None,
                    step=1,
                    repeat=False,
                    mode=None,
                    limit=None,
                    onehot=True):
        
    cur = clip_collection.find()
    
    if skip:
        cur.skip(skip)
    
    if limit:
        cur.limit(limit)
        
    while cur.alive:
        
        yi = []
        
        for _ in range(batch_size):
            # get the next clip
            try:
                clip = get_next_clip(cur, step, repeat, skip, limit)
            
            # abort loop and yield what there is so far
            # this will be the last yield
            except StopIteration:
                break

            # if next clip is fetched successfully 
            # append its info to the lists
            else:
                yi.append(id_from_char[clip['character']])

        if yi == []:
            raise StopIteration
        
        if onehot:
            yield tf.one_hot(yi, 26)
        else:
            yield yi
            