'''
Author : Zack Magnotti
Email : zack@magnotti.net
Date : 3/19/2021

Python module to create a dataset of "clips" -
labeled examples of controller signals of a
specified length saved as pickle files -
from a directory of slippi replay files.
'''

import pickle
from os import path, listdir, makedirs
from random import random

import errno
from slippi.parse import ParseError

from .util import display_progress
from .extract import extract, InvalidGameError, GameTooShortError

class ClippifyFailureError(ValueError):
    '''
    Error class for unknown errors in the clippify process
    '''
    pass

def clippify_game(
        input_filepath,
        output_directory,
        clip_length = 30,
        train_test_split = False,
        current_clip_total = 0
    ):
    '''
    Clippifies a single game file and 
    deposits clips in output_directory.
    
    If train_test_split is true, clips 
    will be randomly split between
    output_directory/train and output_directory/test.

    Parameters
    -----------
    input_filepath (string) : filepath of .slp file
    output_directory (string) : directory to deposit clips
    clip_length (int or float) : length of clips in seconds
    train_test_split (bool) : whether or not to perform train/test split during process
    '''
            
    # get data from game
    player_documents = extract(input_filepath)

    # for counting how many clips came from this game
    game_clip_total = 0
    game_clip_failures = 0

    # iterate over each player
    for doc in player_documents:

        # get istream and metadata
        full_game_istream = doc['istream']
        character = doc['character']
        code = doc['code']
        name = doc['name']
        game_id = doc['game_id']
        
        # clip starting frame (first clip starts at frame 0)
        f = 0              

        # clip length in frames
        step = int(clip_length * 60)          

        while f+step < full_game_istream.shape[0]:
            try:
                # get clip from full game istream
                clip_istream = full_game_istream[f:f+step]
                clip_id = current_clip_total + game_clip_total

                # construct clip payload
                clip_payload = {
                    'istream': clip_istream,
                    'character': character,
                    'name': name,
                    'code': code,
                    'clip_id' : clip_id,
                    'game_id': game_id,
                } 

                # construct clip filename
                clip_filename = f'{character}-{code}-{clip_id}.pkl'

                # construct clip filepath
                if train_test_split:
                    if random() < .1:
                        clip_filepath = path.join(output_directory, 'test', clip_filename)
                    else:
                        clip_filepath = path.join(output_directory, 'train', clip_filename)
                else:
                    clip_filepath = path.join(output_directory, clip_filename)

                # pickle whole document and save to disk
                pickle.dump(clip_payload, open(clip_filepath, 'wb'))

            # if there is a problem with this clip
            except: 
                game_clip_failures += 1
            
            # if clip is saved successfully
            else:
                game_clip_total += 1

            # go to next clip
            finally:
                f += step

    if game_clip_total == 0:
        raise ClippifyFailureError

    return game_clip_total, game_clip_failures

def clippify(
        input_directory,
        output_directory,
        clip_length = 30,
        train_test_split = False
    ):
    ''' 
    Chops up all of the istreams from all of the 
    slp files in input_directory and chops them into clips
    of a given length and deposits the clips into output_directory.

    Filename pattern for clips is:

    {clip_id}-{player_code}-{character}.pkl

    Files are pickled dictionaries containing:
    {
        clip_istream,
        character,
        name,
        code,
        clip_id,
        game_id,
    }

    Parameters
    -----------
    input_directory (string) : directory of .slp files from which to make clips
    output_directory (string) : directory to deposit clips
    clip_length (int or float) : length of clips in seconds
    train_test_split (bool) : whether or not to perform train/test split
    '''

    # normalize paths, list and count files
    input_directory = path.abspath(input_directory)
    output_directory = path.abspath(output_directory)
    file_list = listdir(input_directory)
    N = len(file_list)

    # error checking
    if not path.exists(input_directory):
        raise PathError('input path does not exist')
    
    if not path.isdir(input_directory):
        raise PathError('input path is not a directory')

    # create output directories if they don't exist
    if train_test_split:
        train_dir = path.join(output_directory, 'train')
        test_dir = path.join(output_directory, 'test')
        dirs_to_create = [train_dir, test_dir]
    else:
        dirs_to_create = [output_directory]

    for d in dirs_to_create:
        try: makedirs(d)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # for error tracking
    parse_errors = 0
    games_too_short = 0
    invalid_games = 0
    failed_uploads = 0
    unknown_errors = 0
    clippify_failures = 0
    wrong_filetype = 0
    successful_uploads = 0

    # keep track of how many clips are produced
    clip_total = 0

    # iterate over replay files
    for i, f in enumerate(file_list):

        filepath = path.join(input_directory, f)

        # if filepath is not a slippi file, skip it
        if not path.isfile(filepath):
            wrong_filetype += 1
            continue
        if not path.splitext(filepath)[1] == '.slp':
            wrong_filetype += 1
            continue

        # create clips from replay file and save them to disk
        try: 
            new_clips, new_clippify_failures = clippify_game(
                input_filepath = filepath,
                output_directory = output_directory,
                clip_length = clip_length,
                train_test_split = train_test_split,
                current_clip_total = clip_total
            )
            
        except GameTooShortError:
            failed_uploads += 1
            games_too_short += 1

        except ParseError:
            failed_uploads += 1
            parse_errors += 1

        except InvalidGameError:
            failed_uploads += 1
            invalid_games += 1
        
        except:
            failed_uploads += 1
            unknown_errors += 1

        else:
            successful_uploads += 1
            clip_total += new_clips
            clippify_failures += new_clippify_failures

        finally: # progress bar
            display_progress(i, N)
    display_progress(N,N)

    # Display message after upload is complete
    msg = f'''
Successfully converted {successful_uploads} games into {clip_total} clips.\nFailed to convert {failed_uploads} games.\n
    - {games_too_short} were too short.
    - {parse_errors} failed to parse.
'''

    # if any games were rejected by extract function, display this
    if invalid_games > 0:
        msg += f'    - {invalid_games} were rejected by extract function.\n'

    # if any clippify failures, display this
    if clippify_failures > 0:
        msg += f'    - {clippify_failures} clippify failures.\n'

    # if any files had the wrong extension display this
    if wrong_filetype > 0:
        msg += f'    - {wrong_filetype} paths were not .slp files.\n'

    # if any files had the wrong extension display this
    if unknown_errors > 0:
        msg += f'    - {unknown_errors} unknown errors.\n'

    print(msg)
