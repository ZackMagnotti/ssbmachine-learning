from pymongo import MongoClient
from bson.binary import Binary
from scipy import sparse
from sys import stdout
import os
import pickle

from .extract import extract

from slippi.parse import ParseError

class PathError(ValueError):
    pass

# for convenience to see progress
# when exporting large directories
def display_progress(current_iter, total):

    bar_length = 20

    progress = (bar_length * current_iter) // total
    progress_percent =  round(100 * current_iter / total, 2)

    progress_bar = ('#' * progress) 
    progress_bar += ('.' * (bar_length - progress))
    progress_bar = '[' + progress_bar + ']'

    # return progress_bar

    stdout.write('\r' + progress_bar + ' ' + f'{current_iter} of {total}' + ' - ' + str(progress_percent) + '% ')
    stdout.flush()

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

    file_list = os.listdir(dir_path)
    N = len(file_list)
    num_failed_uploads = 0
    for i, f in enumerate(file_list):

        filepath = os.path.join(dir_path, f)

        if not os.path.splitext(filepath)[-1] == '.slp':
            continue

        try:
            export(f = filepath, 
                   database_name = database_name, 
                   collection_name = collection_name,
                   host = host,
                   port = port)
        except:
            num_failed_uploads += 1
        
        # progress bar
        display_progress(i, N)
    display_progress(N,N)
    print(f'\nFailed to upload {num_failed_uploads} files')

if __name__ == '__main__':
    """
        TODO
        
        If this file is run directly, 
        take arguments using argparse
        and run export() once
    """
    pass
