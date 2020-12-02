from slippi import Game
from slippi.parse import ParseError
from scipy.sparse import lil_matrix, csr_matrix
from os.path import basename
import numpy as np


class InvalidGameError(ValueError):
    pass

class GameTooShortError(ValueError):
    pass

def get_istreams(game, as_sparse=False):
    ''' 
    Gets the controller input streams from a game

    Parameters
    -----------
    game (slippi.Game) : game to get istreams from
    as_sparse (bool) : If true, return istream as a scipy csr matrix
                        otherwise return as numpy array

    Returns
    --------
    istreams (tuple) : 4-tuple representing the controller
                       inputs from every controller port.
                       Inactive ports return as None.
    '''

    istreams = []

    # loop over all 4 controller ports
    for j in range(4):

        # using scipy.sparse.lil_matrix to be
        # efficient incrementally filling in entries row by row
        istream = lil_matrix((len(game.frames), 13))

        # Rows: frames (time)
        # Cols: buttons/joysticks

        for i, frame in enumerate(game.frames):

            # if this port is empty (no controller plugged in)
            if frame.ports[j] is None:
                istream = None
                break
            
            port = frame.ports[j].leader.pre

            # for the analog inputs extract x and y pos
            # (l and r for the triggers)
            istream[i, 0] = port.joystick.x
            istream[i, 1] = port.joystick.y
            istream[i, 2] = port.cstick.x
            istream[i, 3] = port.cstick.y
            istream[i, 4] = port.triggers.physical.l
            istream[i, 5] = port.triggers.physical.r
            
            # for the digital inputs, fill in with 1
            # if button is active (leave as 0 if not)
            b = port.buttons
            if b.Physical.Y in b.physical.pressed():
                istream[i, 6] = 1
            
            if b.Physical.X in b.physical.pressed():
                istream[i, 7] = 1
            
            if b.Physical.B in b.physical.pressed():
                istream[i, 8] = 1
            
            if b.Physical.A in b.physical.pressed():
                istream[i, 9] = 1
            
            if b.Physical.L in b.physical.pressed():
                istream[i, 10] = 1
            
            if b.Physical.R in b.physical.pressed():
                istream[i, 11] = 1
            
            if b.Physical.Z in b.physical.pressed():
                istream[i, 12] = 1

        # if as_sparse is true and port is active, 
        # convert to compressed sparse array
        # else convert to numpy array
        else:
            if as_sparse:
                istream = csr_matrix(istream)
            else:
                istream = istream.toarray()
        istreams.append(istream)
    
    # len(istreams) == 4
    # each element represents istream for 
    # one of the four controller ports (players 1-4)
    # empty ports give an 'istream' of None
    return tuple(istreams)

def get_player_characters(game):
    ''' 
    Gets the player characters from a game

    Parameters
    -----------
    game (slippi.Game) : game to get player characters from

    Returns
    --------
    characters (tuple) : 4-tuple representing character
                         selections from every controller port.
                         Inactive ports return as None.
    '''

    players = game.start.players

    characters = [None]*4
    for i, player in enumerate(players):
        if player is not None:
            character = player.character
            characters[i] = character.name

    return tuple(characters)

def get_player_names(game):
    ''' 
    Gets the player names from a game

    Player names are provided by metadata that may not be 
    present for games played on offline platforms. 

    Parameters
    -----------
    game (slippi.Game) : game to get player names from

    Returns
    --------
    characters (tuple) : 4-tuple representing player names
                         from every controller port.
                         Inactive ports return as None.
    '''

    players = game.metadata.players

    names = [None]*4
    for i, player in enumerate(players):
        if player and player.netplay:
            names[i] = player.netplay.name

    return tuple(names)

def get_player_codes(game):
    ''' 
    Gets the player netplay codes from a game

    Player codes are provided by metadata that may not be 
    present for games played on offline platforms. 

    Parameters
    -----------
    game (slippi.Game) : game to get player codes from

    Returns
    --------
    characters (tuple) : 4-tuple representing player codes
                         from every controller port.
                         Inactive ports return as None.
    '''

    players = game.metadata.players

    codes = [None]*4
    for i, player in enumerate(players):
        if player and player.netplay:
            codes[i] = player.netplay.code

    return tuple(codes)

def get_id(f):
    ''' 
    Gets the game_id for a particular replay file.

    The game_id is just the filename of that game.
    This has the advantage of being unique, as
    long as each collection only contains
    games from a single directory

    Parameters
    -----------
    f (string) : Full path to game replay file

    Returns
    -----------
    game_id (string) : the (uniqe) filename of the input filepath
    '''

    return basename(f)

def extract(f, as_sparse=False): 
    ''' 
    Extracts the istream payloads from a .slp file

    Parameters
    -----------
    f (string) : Full path to game replay file
    as_sparse (bool) : If true, return istream as a scipy csr matrix
                        otherwise return as numpy array

    Returns
    -----------
    payload (tuple of dicts) : returns the player data from the replay file
                               for all players detected.
                               Each of these dicts is essentially the json
                               object that will be sent to mongodb in export.py
    '''

    # get game_id and game data using f (filename)
    game_id = get_id(f)
    game = Game(f)

    # reject games less than a minute long
    if len(game.frames)/3600 < 1:
        raise GameTooShortError('Game is too short')

    # get outpt payload for each active controller port
    try:
        payload = [
        {
            'game_id': game_id,
            'istream': istream,
            'character': character,
            'name': name,
            'code': code,
        } 
        for istream, character, name, code
            in zip(get_istreams(game, as_sparse=as_sparse), 
                   get_player_characters(game),
                   get_player_names(game),
                   get_player_codes(game))
            if character is not None
        ]

    # raise parsing errors
    except ParseError:
        raise
    
    # if unexpected error, raise InvalidGameError
    except:
        raise InvalidGameError
        
    else:
        return tuple(payload)
