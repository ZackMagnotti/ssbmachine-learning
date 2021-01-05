from pymongo import MongoClient
from bson.binary import Binary
from scipy.sparse import csr_matrix
import numpy as np
import pickle

from .export import display_progress

def clippify(input_collection,
             output_collection,
             clip_length = 30,
             max_clips = None,
             query = {}):

    if max_clips is None:
        N = input_collection.estimated_document_count()

    clip_count = 0 # each clip will get a number for indexing use

    cursor = input_collection.find(query)
    for i, doc in enumerate(cursor):
            
        try:
            # chop every game up into clips
            full_game_istream = pickle.loads(doc['istream'])
            clip_istreams = []
            F = clip_length * 60 # clip length in frames
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
            pass
        
        else:
            output_collection.insert_many(payload)
            clip_count += len(payload)
        
        finally:
            if max_clips and clip_count > max_clips:
                return

        if max_clips is None:
            display_progress(i, N)
        else:
            display_progress(clip_count, max_clips)

    if max_clips is None:
        display_progress(N, N)
    else:
        display_progress(max_clips, max_clips)
