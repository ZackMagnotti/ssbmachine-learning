from pymongo import MongoClient
import numpy as np
import tensorflow as tf
import pickle

from src.util import characters, id_from_char, char_from_id

def get_next_clip(cur, step, repeat, skip, limit):
    '''
        advance the cursor step times and get
        that clip
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
        
        # get the ntext batch
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
                yi.append(id_from_char[clip['character']])

        if xi == []:
            raise StopIteration
        
        xi = np.stack(xi, axis=0)
        
        if onehot:
            yi = tf.one_hot(yi, 26)

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
        
        # get the ntext batch
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
        
        # get the ntext batch
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
            