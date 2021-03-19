from sys import stdout

def display_progress(i, N):
    ''' 
    For convenience, to see progress
    when exporting large directories.

    Parameters
    -----------
    i (int) : current iteration
    N (int) : total number of iterations in process
    '''

    bar_length = 20

    progress = int((bar_length * i) // N)
    progress_percent =  round(100 * i / N, 2)

    progress_bar = ('#' * progress)
    progress_bar += ('.' * (bar_length - progress))
    progress_bar = '[' + progress_bar + ']'

    stdout.write(f'\r{progress_bar} {i} of {N} - {progress_percent}% ')
    stdout.flush()

characters = (
    'CAPTAIN_FALCON',
    'DONKEY_KONG',
    'FOX',
    'GAME_AND_WATCH',
    'KIRBY',
    'BOWSER',
    'LINK',
    'LUIGI',
    'MARIO',
    'MARTH',
    'MEWTWO',
    'NESS',
    'PEACH',
    'PIKACHU',
    'ICE_CLIMBERS',
    'JIGGLYPUFF',
    'SAMUS',
    'YOSHI',
    'ZELDA',
    'SHEIK',
    'FALCO',
    'YOUNG_LINK',
    'DR_MARIO',
    'ROY',
    'PICHU',
    'GANONDORF',
)

id_from_char = {c:i for i, c in enumerate(characters)}

char_from_id = {i:c for i, c in enumerate(characters)}
