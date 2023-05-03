import json
#import requests
from os.path import exists

import openai

from gpt_utils import *

SECRETS = 'secrets.json'

if exists(SECRETS):
    with open(SECRETS) as f:
        api_key = json.load(f)['api_key']
else:
    raise FileNotFoundError(f'''No file {SECRETS} found
    If you\'re trying to run this, you need to get an OpenAI API token
    and put it in a json file called {SECRETS} with the field \'api_key\'''')

test_prompt = "Context: China suspended initial public offerings (IPOs) in early July (http://www. bloomberg. com/news/articles/2015-07-04/china-stock-brokers-set-up-19-billion-fund-to-stem-market-rout , http://www. reuters. com/article/2015/07/05/us-china-markets-brokerage-pledge-idUSKCN0PE08E20150705 , http://www. wsj. com/articles/china-setting-up-fund-to-stabilize-stock-market-1435991611 ). \\nQuestion: Will there be an initial public offering on either the Shanghai Stock Exchange or the Shenzhen Stock Exchange before 1 January 2016?\\nChoices: yes\\nno\\n###\\n"

response = openai.Completion.create(
    model='ft-jUzTYEt84vpetUOIrgZCUjpp',
    prompt=test_prompt)


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