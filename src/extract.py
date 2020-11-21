from slippi import Game
import numpy as np

def get_istreams(game): # f: filename of .slp file
    frames = game.frames

    # get istream for each port
    p1_istream = np.zeros((len(frames), 17)) # rows: frames, columns: buttons
    p2_istream = np.zeros((len(frames), 17))
    p3_istream = np.zeros((len(frames), 17))
    p4_istream = np.zeros((len(frames), 17))

    for j, istream in enumerate([p1_istream, p2_istream, p3_istream, p4_istream]):
        for i, frame in enumerate(frames):
            
            if frame.ports[j] is None:
                istream = None
                break
            
            port = frame.ports[j].leader.pre

            istream[i, 0] = port.joystick.x
            istream[i, 1] = port.joystick.y
            istream[i, 2] = port.triggers.physical.l
            istream[i, 3] = port.triggers.physical.r
            
            b = port.buttons
            if b.Physical.Y in b.physical.pressed():
                istream[i, 4] = 1
            
            if b.Physical.X in b.physical.pressed():
                istream[i, 5] = 1
            
            if b.Physical.B in b.physical.pressed():
                istream[i, 6] = 1
            
            if b.Physical.A in b.physical.pressed():
                istream[i, 7] = 1
            
            if b.Physical.L in b.physical.pressed():
                istream[i, 8] = 1
            
            if b.Physical.R in b.physical.pressed():
                istream[i, 9] = 1
            
            if b.Physical.Z in b.physical.pressed():
                istream[i, 10] = 1
            
            if b.Physical.DPAD_UP in b.physical.pressed():
                istream[i, 11] = 1
            
            if b.Physical.DPAD_DOWN in b.physical.pressed():
                istream[i, 12] = 1
            
            if b.Physical.DPAD_RIGHT in b.physical.pressed():
                istream[i, 13] = 1
            
            if b.Physical.DPAD_LEFT in b.physical.pressed():
                istream[i, 14] = 1
            
            istream[i, 15] = port.cstick.x
            istream[i, 16] = port.cstick.y
    
    return (p1_istream, p2_istream, p3_istream, p4_istream)

def get_characters(game):
    pass

def get_id(game):
    pass

def get_players(game):
    pass

def extract(f):
    game = Game(f)
    istreams = get_istreams(game)

def export():
    pass