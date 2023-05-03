from ast import literal_eval
import pandas as pd

SOURCE_DIR = 'data/source'
CONTEXT_DIR = 'data/with_context'

# TODO: also add context to test data
data = pd.read_csv(SOURCE_DIR + 'autocast_train.csv')

for datum in data:
    links = literal_eval(datum['source_links'])
    for link in links:
        pass # TODO: could use https://github.com/andyzoujm/autocast/blob/master/autocast_experiments/data/ at the cost of many more tokens