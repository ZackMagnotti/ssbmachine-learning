from slippi import Game
from scipy.sparse import lil_matrix, csr_matrix
from os.path import basename
import numpy as np

'''
    TODO: Docstrings
'''

class InvalidGameError(ValueError):
    pass

class EmptyFilenameError(ValueError):
    pass

class GameTooShortError(ValueError):
    pass

def get_istreams(game, as_sparse=False):

    istreams = []

    # loop over all 4 controller ports
    for j in range(4):

        # using scipy.sparse.lil_matrix to be
        # efficient filling in entries incrementally
        istream = lil_matrix((len(game.frames), 13))

        for i, frame in enumerate(game.frames):

            # if this port is empty (no controller plugged in)
            if frame.ports[j] is None:
                istream = None
                break
            
            port = frame.ports[j].leader.pre

            # each column represents a button
            istream[i, 0] = port.joystick.x
            istream[i, 1] = port.joystick.y
            istream[i, 2] = port.cstick.x
            istream[i, 3] = port.cstick.y
            istream[i, 4] = port.triggers.physical.l
            istream[i, 5] = port.triggers.physical.r
            
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

            # if b.Physical.DPAD_UP in b.physical.pressed():
            #     istream[i, 13] = 1
            
            # if b.Physical.DPAD_DOWN in b.physical.pressed():
            #     istream[i, 14] = 1
            
            # if b.Physical.DPAD_RIGHT in b.physical.pressed():
            #     istream[i, 15] = 1
            
            # if b.Physical.DPAD_LEFT in b.physical.pressed():
            #     istream[i, 16] = 1

        # if as_sparse is true,
        # convert to compressed sparse array
        if as_sparse and istream is not None:
            istream = csr_matrix(istream)
        # else convert to numpy array
        elif istream is not None:
            istream = istream.toarray()
        istreams.append(istream)
    
    # len(istreams) == 4
    # each element represents istream for 
    # one of the four controller ports (players 1-4)
    # empty ports give an 'istream' of None
    return tuple(istreams)

def get_player_characters(game):
    players = game.start.players

    characters = [None]*4
    for i, player in enumerate(players):
        if player is not None:
            character = player.character
            characters[i] = character.name

    return tuple(characters)

def get_player_names(game):
    # player name relies on metadata that may not be there
    # as a result dataset may have many missing names
    players = game.metadata.players

    names = [None]*4
    for i, player in enumerate(players):
        if player and player.netplay:
            names[i] = player.netplay.name

    return tuple(names)

def get_player_codes(game):
    # player code relies on metadata that may not be there
    # as a result dataset may have many missing codes
    players = game.metadata.players

    codes = [None]*4
    for i, player in enumerate(players):
        if player and player.netplay:
            codes[i] = player.netplay.code

    return tuple(codes)

def get_id(f):
    # game_id is just the filename of that game.
    # This has the advantage of being unique, as
    # long as each collection only contains
    # games from a single directory

    return basename(f)

def extract(f, as_sparse=False): 
    game_id = get_id(f)
    game = Game(f)

    # reject games less than a minute long
    if len(game.frames)/3600 < 1:
        raise GameTooShortError('Game is too short')

    try:
        out = [{'game_id': game_id,
                'istream': istream,
                'character': character,
                'name': name,
                'code': code,
        } for istream, character, name, code
            in zip(get_istreams(game, as_sparse=as_sparse), 
                   get_player_characters(game),
                   get_player_names(game),
                   get_player_codes(game))
            if character is not None]
    except:
        raise InvalidGameError
        
    return tuple(out)

if __name__=='__main__':
    pass
