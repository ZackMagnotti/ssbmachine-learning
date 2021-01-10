from slippi.parse import ParseError
from bson.binary import Binary
from pymongo import MongoClient
from scipy import sparse
from sys import stdout
from os import path, listdir
import pickle
from .extract import extract, InvalidGameError, GameTooShortError

from .util import display_progress

class PathError(ValueError):
    pass

def export(f, 
           database_name, 
           collection_name,
           host = 'localhost',
           port = 27017,
           client = None):
    ''' 
    Extracts the istream payloads from a .slp file and
    exports them to the specified mongoDB database.collection

    Parameters
    -----------
    f (string) : Full path to game replay file
    database_name (string) : name of mongo database
    collection_name (string) : name of collection in database
    host : see py-mongo documentation
    port (int) : port number on which to connect
    client (optional) : to use existing client. If None, a new connection will be made.
    '''

    # Connect to the hosted MongoDB instance
    if client is None:
        client = MongoClient(host, port)
    db = client[database_name]
    collection = db[collection_name]
    
    # extract payload data from file using extract.py
    payloads = extract(f)

    # pickle istream data before sending to mongo db
    mongo_output = []
    for payload in payloads:
        pickled = {}
        for k, v in payload.items():
            # if value is a sparse matrix, convert to binary
            if isinstance(v, sparse.csr.csr_matrix):
                pickled[k] = Binary(pickle.dumps(v, protocol=2))
            else:
                pickled[k] = v
        
        # export data to mongodb
        mongo_output.append(pickled)
    collection.insert_many(mongo_output)

def export_dir(dir_path, 
               database_name, 
               collection_name,
               host = 'localhost',
               port = 27017,
               client = None):
    ''' 
    Extracts the istream payloads from a directory of .slp files and
    exports them to the specified mongoDB database.collection

    Parameters
    -----------
    dir_path (string) : Full path to directory of replay files
    database_name (string) : name of mongo database
    collection_name (string) : name of collection in database
    host : see py-mongo documentation
    port (int) : port number on which to connect
    '''

    if client is None:
        client = MongoClient(host, port)
    dir_path = path.normpath(dir_path)
    file_list = listdir(dir_path)
    N = len(file_list)
    
    if not path.exists(dir_path):
        raise PathError('input path does not exist')
    
    if not path.isdir(dir_path):
        raise PathError('input path is not a directory')

    # for error tracking
    num_parse_errors = 0
    num_games_too_short = 0
    num_invalid_games = 0
    num_failed_uploads = 0
    num_successful_uploads = 0

    for i, f in enumerate(file_list):

        filepath = path.join(dir_path, f)

        # if filepath is not a slippi file, skip it
        if not path.isfile(filepath):
            continue
        if not path.splitext(filepath)[1] == '.slp':
            continue

        try:
            export(f = filepath, 
                   database_name = database_name, 
                   collection_name = collection_name,
                   client = client)

        except GameTooShortError:
            num_failed_uploads += 1
            num_games_too_short += 1

        except ParseError:
            num_failed_uploads += 1
            num_parse_errors += 1

        except InvalidGameError:
            num_failed_uploads += 1
            num_invalid_games += 1

        else:
            num_successful_uploads += 1

        finally:
            display_progress(i, N)
    display_progress(N,N)

    # Display message after upload is complete
    msg = f'''\nSuccessfully uploaded {num_successful_uploads} games.\nFailed to upload {num_failed_uploads} games.\n
    - {num_games_too_short} were too short.
    - {num_parse_errors} failed to parse.\n'''

    # if any games were rejected by extract function, display this
    if num_invalid_games > 0:
        msg += f'    - {num_invalid_games} were rejected by extract function\n'

    print(msg)
