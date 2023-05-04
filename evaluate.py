import pandas as pd
import json

from config import SOURCE_DIR, FINETUNED_DIR

model_out_path = FINETUNED_DIR + 'completions.json'

with open(model_out_path, 'r') as f:
    completions = json.load(f)

train_data = pd.read_csv(SOURCE_DIR + 'autocast_train.csv')

answers = train_data['answers']

