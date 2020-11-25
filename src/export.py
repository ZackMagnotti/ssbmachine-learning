from pymongo import MongoClient
from bson.binary import Binary
from scipy import sparse
import pickle

from extract import extract

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

if __name__ == '__main__':
    """
        TODO
        
        If this file is run directly, 
        take arguments using argparse
        and run export() once
    """
    pass
