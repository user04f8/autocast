import pandas as pd

from gpt_utils import get_prompt

SOURCE_DIR = 'data/source/'
FINETUNED_DIR = 'data/gpt_finetune/'

data = pd.read_csv(SOURCE_DIR + 'autocast_train.csv')

prompts = [get_prompt(q, b, c) for q, b, c in zip(data['question'], data['background'], data['qtype'], data['choices'])]
ideal_responses = data['answer']


print(prompts[:3])