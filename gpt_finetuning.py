import pandas as pd
import json

from gpt_utils import *

SOURCE_DIR = 'data/source/'
FINETUNED_DIR = 'data/gpt_finetune/'

for train in (True, False):
    if train:
        in_file = SOURCE_DIR + 'autocast_train.csv'
        out_file = FINETUNED_DIR + 'data_train.json'
    else:
        in_file = SOURCE_DIR + 'autocast_test.csv'
        out_file = FINETUNED_DIR + 'data_test.json'
    data = pd.read_csv(in_file)

    if train:
        data = data.loc[data['status'] == 'Resolved']
        # strip unresolved data

    print('total resolved data pts:', len(data))

    qtypes = {'t/f': get_prompt_t_f, 'mc': get_prompt_multi, 'num': get_prompt_numeric}

    with open(out_file, 'w') as f:
        f.write('')
    with open(out_file, 'a') as f:
        if train:
            for qtype, get_prompt in qtypes.items():
                data_type = data.loc[data['qtype'] == qtype]
                print(f'preprocessed data for {qtype} with {len(data_type)} datums')
                prompts_and_completions = [get_prompt(clean(q), b, c, a) for q, b, c, a in zip(data_type['question'], data_type['background'], data_type['choices'], data_type['answer'])]

                for prompt, ideal_response in prompts_and_completions:
                    d = {"prompt": prompt, "completion": ideal_response} #.replace('"', '\\"')
                    line = json.dumps(d)
                    #print(line)
                    f.write(line + '\n')
                #f.write('\n'.join(json.dumps({"prompt": prompt.replace('"', '\\\"'), "completion": ideal_response.replace('"', '\\\"')}) for prompt, ideal_response in prompts_and_completions)) 
                # f'{{"prompt": " {prompt}", "completion": " {ideal_response} "}}'
        else:
            prompt_list = []
            for qtype, get_prompt in qtypes.items():
                data_type = data.loc[data['qtype'] == qtype]
                print(f'preprocessed data for {qtype} with {len(data_type)} datums')
                prompt_list += [get_prompt(clean(q), b, c) for q, b, c in zip(data_type['question'], data_type['background'], data_type['choices'])]
            json.dump({'prompt_list': prompt_list}, f)
    #with open(out_file, 'r', encoding='utf-8') as f:
    #    for line in f:
    #        data = json.loads(line)
    #        prompt = data['prompt']
    #        ideal_response = data['completion']
            # TEST