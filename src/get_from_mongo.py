from pymongo import MongoClient
# import numpy as np
# import pickle

"""
    TODO: finish this or remove unused parameters
"""

def get_data(database_name, 
             collection_name,
             host = 'localhost',
             port = 27017,
             client = None,
             max_results = None,
             get = None,
             query = {}):
    '''
        Gets data from the specified mongo collection

        Parameters
        ----------
        database_name (string)
        collection_name (string)
        host (optional)
        port (int/optional)
        client (optional)
        get (int|float) : indicates how many results to fetch.
                        If this is an int, get that many results.
                        If this is a float between 0 and 1, return
                        that proportion of the total results.
        query (keyword pairs): additional search requirements if applicable

        Returns
        -------
        X (list of sparse matrices): the input data for each document
        y (array of strings): the character selection for each document

        X, y = get_data(...)
    '''

    # Connect to the hosted MongoDB instance
    if client is None:
        client = MongoClient(host, port)
    db = client[database_name]
    collection = db[collection_name]

    characters = []
    istreams = []
    for player in collection.find(query):
        istreams.append(player['istream'])
        characters.append(player['character'])

    return istreams, characters
