from typing import Tuple
from datetime import datetime, timedelta
from ast import literal_eval

SEPARATOR = '\\n###\\n'
stop_sequence_token = '###'
STOP_SEQUENCE = ' ###'

#multi_choice_tokens ('A' ...)
num_choice_tokens = tuple(chr(n) for n in range(65, 65 + 11))
num_split_len = len(num_choice_tokens) - 1



def clean(s : str):
    return s.replace('”', '"').replace('“', '"').replace('"', '\\\"')

def inv_numeric(answer_chr : str) -> float:
    return (ord(answer_chr) - 65) / num_split_len

def get_prompt_numeric(question : str, background : str, choices : str, answer=None) -> Tuple[str, str]:
    """
    Returns a tuple with a prompt and ideal response for GPT finetuning, specific to numeric questions
    """
    choices = literal_eval(choices)
    if isinstance(choices['max'], str):
        max_date = datetime.strptime(choices['max'], '%Y-%m-%d')
        low = min_date = datetime.strptime(choices['min'], '%Y-%m-%d')
        range = (max_date - min_date).days
        choices_text = '\\n'.join(f'{token}: {min_date + timedelta(days=((i/num_split_len) * range))}' for i, token in enumerate(num_choice_tokens))
        #completion_raw : datetime = min_date + timedelta(days=(range ** answer))
        #completion = completion_raw.strftime('%Y-%m-%d')
    else:
        range = choices['max'] - choices['min']
        low = choices['min']
        choices_text = '\\n'.join(f'{token}: {low + (((i/num_split_len)) * range)}' for i, token in enumerate(num_choice_tokens))
    #choices['deriv_ratio']
    prompt = f'Context: {background}\\nQuestion: {question}\\nWhich choice is closest?\\n{choices_text}{SEPARATOR}'

    if answer is None:
        return prompt
    else:
        answer = float(answer) # from 0. to 1. (log scaled)
        if answer == 2.0:
            # WHY? out of range annoying data points; there's only two outliers though so we'll just treat them as == 1.0
            answer = 1.0
        else:
            assert 0. <= answer <= 1., f'{answer}'
        n = int(answer * num_split_len)
        
        if n > num_split_len:
            print(n, question, choices, range, answer)
        try:
            completion = f'{num_choice_tokens[n]} or exactly {answer:.4f}{STOP_SEQUENCE}'
        except:
            
            print(n, question, choices, range, answer)
            return None
    return prompt, completion


def get_prompt_t_f(question : str, background : str, choices : str, answer=None) -> Tuple[str, str]:
    """
    Returns a tuple with a prompt and ideal response for GPT finetuning
    """
    try:
        choices = literal_eval(choices)
        choices_text = "\\n".join(choices)

        prompt = f'Context: {background}\\nQuestion: {question}\\nChoices: {choices_text}{SEPARATOR}'
        if answer is None:
            return prompt
        else:
            return prompt, answer + STOP_SEQUENCE

    except Exception as e:
        print(f'warn on prompt {question} {background} {choices}')
        print(e)
        
        return None
    
def get_prompt_multi(question : str, background : str, choices : str, answer=None) -> Tuple[str, str]:
    """
    Returns a tuple with a prompt and ideal response for GPT finetuning
    """
    try:
        choices = literal_eval(choices)
        choices_text = "\\n".join(choices) # could do A: ..., B: ..., etc.
        prompt = f'Context: {background}\\nQuestion: {question}\\nChoices:\\n{choices_text}{SEPARATOR}'
        if answer is None:
            return prompt
        else:
            completion = choices[ord(answer) - 65] + STOP_SEQUENCE
            return prompt, completion

    except Exception as e:
        print(f'warn on prompt {question} {background} {choices}')
        print(e)

        return None
    
qtypes = {'t/f': get_prompt_t_f, 'mc': get_prompt_multi, 'num': get_prompt_numeric}

def get_prompt(question : str, background : str, choices : str, qtype):
    funct = qtypes[qtype]
    return funct(question, background, choices)
