'''
    Depreciated Code

    Was useful when dataset was in mongodb,
    and I needed to get it out.

    Now data pipeline avoids the use of mongodb
    entirely, so this code is only needed when 
    converting old versions of the dataset.
'''

# from os.path import join, splitext
# from random import random

# for i, doc in enumerate(cursor):
#     # get metadata / labels
#     character = doc['character']
#     code = doc['code']
#     game_id = doc['game_id']
#     clip_id = doc['clip_id']

#     # construct filename
#     filename = f'{character}-{code}-{clip_id}.pkl'
# #     filename = f'{clip_id}_{character}_{code}_{game_id}.pkl'
    
#     if random() < .1:
#         filepath = join(parentdir, 'data', 'character','test', filename)
#     else:
#         filepath = join(parentdir, 'data', 'character', 'train', filename)
        
#     # unpickle istream
#     doc['istream'] = pickle.loads(doc['istream'])

#     # pickle whole document
#     pickle.dump(doc, open(filepath, 'wb'))
    
#     if i % 100 == 0:
#         display_progress(i, N)
# display_progress(N, N)