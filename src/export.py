from bson.binary import Binary
from pymongo import MongoClient
from scipy import sparse
from sys import stdout
from os import path, listdir
import pickle
from .extract import extract, InvalidGameError, GameTooShortError

'''
    TODO: Docstrings
'''

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

    # compress istream data before sending to mongo db
    mongo_output = []
    for player in players:
        compressed = {}
        for k, v in player.items():
            if isinstance(v, sparse.csr.csr_matrix):
                # if value is a sparse matrix, convert to binary
                compressed[k] = Binary(pickle.dumps(v, protocol=2))
            else:
                compressed[k] = v
        
        # export data to mongodb
        mongo_output.append(compressed)
    collection.insert_many(mongo_output)

def export_dir(dir_path, 
               database_name, 
               collection_name,
               host = 'localhost',
               port = 27017):

    dir_path = path.normpath(dir_path)
    
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

    file_list = listdir(dir_path)
    N = len(file_list) # for progress bar
    for i, f in enumerate(file_list): #enumerate is for progress bar

        filepath = path.join(dir_path, f)

        # if filepath is not a slippi file, skip it
        if not path.isfile(filepath):
            continue
        if not path.splitext(filepath)[-1] == '.slp':
            continue

        try:
            export(f = filepath, 
                   database_name = database_name, 
                   collection_name = collection_name,
                   host = host,
                   port = port)

            num_successful_uploads += 1

        except GameTooShortError:
            num_failed_uploads += 1
            num_games_too_short += 1

        except ParseError:
            num_failed_uploads += 1
            num_parse_errors += 1

        except InvalidGameError:
            num_failed_uploads += 1
            num_invalid_games += 1
        
        # progress bar
        display_progress(i, N)
    display_progress(N,N)

    msg = f'''\nSuccessfully uploaded {num_successful_uploads} games.\nFailed to upload {num_failed_uploads} games.\n
    - {num_games_too_short} were too short.
    - {num_parse_errors} failed to parse.\n'''
    # if any games were rejected by extract function, display this
    if num_invalid_games > 0:
        msg += f'    - {num_invalid_games} were rejected by extract function\n'

    print(msg)
