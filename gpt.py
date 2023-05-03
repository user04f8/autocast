import json
import requests
from os.path import exists

SECRETS = 'secrets.json'

if exists(SECRETS):
    with open(SECRETS) as f:
        api_key = json.load(f)['api_key']
else:
    raise FileNotFoundError(f'''No file {SECRETS} found
    If you\'re trying to run this, you need to get an OpenAI API token
    and put it in a json file called {SECRETS} with the field \'api_key\'''')

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