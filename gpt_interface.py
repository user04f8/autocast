import json
#import requests
from os.path import exists
import json

import openai

import gpt_utils

SECRETS = 'secrets.json'
FINETUNED_MODEL = 'curie:ft-personal-2023-05-03-09-26-57'
# generate a new model with:
# openai api fine_tunes.create -t data\gpt_finetune\data_train.json -m curie

if exists(SECRETS):
    with open(SECRETS) as f:
        openai.api_key = json.load(f)['api_key']
else:
    raise FileNotFoundError(f'''No file {SECRETS} found
    If you\'re trying to run this, you need to get an OpenAI API token
    and put it in a json file called {SECRETS} with the field \'api_key\'''')

with open('data/gpt_finetune/data_test.json') as f:
    test_data = json.load(f)
with open('data/gpt_finetune/data_train.json') as f:
    #train_data = json.load(f)
    train_data_lines = f.readlines()

test_datum = json.loads(train_data_lines[3477])
print(test_datum)
test_prompt = test_datum['prompt']

response = openai.Completion.create(
    model=FINETUNED_MODEL,
    prompt=test_prompt,
    max_tokens=8,
    stop=gpt_utils.stop_sequence_token)

print(response)

print(response['choices'][0]['text'])

"""
url = "https://api.openai.com/v1/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

data = {
    "model": "text-davinci-002",
    "prompt": "Correct this to standard English : Who are you \n",
    "max_tokens": 60        
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
"""