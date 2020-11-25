from slippi import Game
from scipy import sparse
import numpy as np

'''
    TODO: Docstrings
'''

class InvalidGameError(ValueError):
    pass

class EmptyFilenameError(ValueError):
    pass

def get_istreams(game, as_sparse=False):

    out = [] # list to store output

    for j in range(4):

        # rows: frames, columns: buttons
        # istream = np.zeros((len(game.frames), 17))
        istream = np.zeros((len(game.frames), 13))

        for i, frame in enumerate(game.frames):
            if frame.ports[j] is None:
                istream = None
                break
            
            port = frame.ports[j].leader.pre

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

        if as_sparse and istream is not None:
            istream = sparse.csr_matrix(istream)
        out.append(istream)
    
    # len(out) == 4
    # each element represents istream
    # for one of the four ports (player 1-4)
    # empty ports give an 'istream' of None
    return tuple(out)

def get_player_characters(game):
    players = game.metadata.players

    characters = [None]*4
    for i, player in enumerate(players):
        if player is not None:
            character = next(iter(player.characters))
            characters[i] = character.name

    return tuple(characters)

def get_player_names(game):
    players = game.metadata.players

    names = [None]*4
    for i, player in enumerate(players):
        if player is not None:
            names[i] = player.netplay.name

    return tuple(names)

def get_id(f):
    if f == '':
        msg = 'path can not be empty string'
        raise EmptyFilenameError(msg)

    return f.replace('\\', '/').split('/')[-1]

def extract(f, as_sparse=False):
    game_id = get_id(f)
    game = Game(f)
    try:
        out = [{'game_id': game_id,
                'istream': istream,
                'character': character,
                'name': name,
        } for istream, character, name 
            in zip(get_istreams(game, as_sparse=as_sparse), 
                   get_player_characters(game),
                   get_player_names(game))
            if character is not None]
    except AttributeError:
        # if netplay information is missing
        # that means the game failed to connect
        # and was aborted
        raise InvalidGameError("This game was aborted")
        
    return tuple(out)
