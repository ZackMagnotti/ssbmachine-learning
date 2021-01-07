from pymongo import MongoClient
from bson.binary import Binary
from scipy.sparse import csr_matrix
import numpy as np
import pickle

from .export import display_progress

def clippify(input_collection,
             output_collection,
             clip_length = 30,
             query = {}):
    ''' 
    Chops up all of the istreams in a collection into clips 
    of a given length and deposits the clips into another collection.

    Parameters
    -----------
    input_collection (string) : collection of istreams from which to make clips
    output_collection (string) : collection to deposit clips
    clip_length (int or float) : length of clips in seconds
    query (dict) : optional query to filter which istreams from input_collection get clippified
    '''

    cursor = input_collection.find(query)
    N = input_collection.estimated_document_count()
    clip_count = 0 # used to index each clip

    failures = 0

    for i, doc in enumerate(cursor):
            
        try:
            # chop every game up into clips
            full_game_istream = pickle.loads(doc['istream'])
            clip_istreams = []
            F = int(clip_length * 60) # clip length in frames
            f = 0      # clip starting frame
            while f+F < full_game_istream.shape[0]:
                clip_istreams.append(full_game_istream[f:f+F])
                f += F

            payload = [
                {
                    'game_id': doc['game_id'],
                    'clip_id' : j + clip_count, 
                    'istream': Binary(pickle.dumps(istream, protocol=2)),
                    'character': doc['character'],
                    'name': doc['name'],
                    'code': doc['code'],
                } 
                for j, istream in enumerate(clip_istreams)
            ]
        
        except:
            failures += 1
        
        else:
            output_collection.insert_many(payload)
            clip_count += len(payload)

        display_progress(i, N)
    display_progress(N, N)

    if failures > 0:
        print(f"\n{failures} istreams failed to clippify\n")
