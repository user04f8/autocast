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

def run_all():
    """
    Execute the completions on each token
    to cache the results in completions.json
    (so that we minimize API calls to openai)
    """
    completions = []
    with open('data/gpt_finetune/completions.txt', 'w') as f:
        for test_q_prompt in test_data['prompt_list']:
            test_q, test_prompt = test_q_prompt
            response = openai.Completion.create(
                model=FINETUNED_MODEL,
                prompt=test_prompt,
                max_tokens=32,
                stop=gpt_utils.stop_sequence_token)
            completion_text = response['choices'][0]['text']
            completions.append({'question': test_q, 'completion': completion_text.strip()})
            f.write(completion_text.strip() + '\n')
            print(completion_text)

    with open('data/gpt_finetune/completions.json', 'w') as f:
        json.dump(completions, f)

def test():
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

if __name__ == '__main__':
    run_all()