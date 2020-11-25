from pymongo import MongoClient
from bson.binary import Binary
from scipy import sparse
import os
import pickle

from .extract import extract

class PathError(ValueError):
    pass

def export(f, 
           database_name, 
           collection_name,
           host = 'localhost',
           port = 27017):
    # Connect to the hosted MongoDB instance
    client = MongoClient(host, port)
    db = client[database_name]
    collection = db[collection_name]
    
    players = extract(f, as_sparse=True)
    mongo_output = []
    for player in players:
        sanitized = {}
        for k, v in player.items():
            if isinstance(v, sparse.csr.csr_matrix):
                # if value is a sparse matrix, convert to binary
                sanitized[k] = Binary(pickle.dumps(v, protocol=2))
            else:
                sanitized[k] = v
        
        # export data to mongodb
        mongo_output.append(sanitized)
    collection.insert_many(mongo_output)

def export_dir(dir_path, 
               database_name, 
               collection_name,
               host = 'localhost',
               port = 27017):

    dir_path = os.path.normpath(dir_path)
    
    if not os.path.exists(dir_path):
        raise PathError('The input path does not exist')
    
    if not os.path.isdir(dir_path):
        raise PathError('The input path is not a directory')

    for f in os.listdir(dir_path):
        filepath = os.path.join(dir_path, f)

        if not os.path.splitext(filepath)[-1] == '.slp':
            continue

        export(f = filepath, 
               database_name = database_name, 
               collection_name = collection_name,
               host = host,
               port = port)

if __name__ == '__main__':
    """
        TODO
        
        If this file is run directly, 
        take arguments using argparse
        and run export() once
    """
    pass
