from typing import Tuple
from datetime import datetime
from ast import literal_eval


num_choice_tokens = ('A', 'B', 'C', 'D', 'E', 'F', 'G')
num_split_len = len(num_choice_tokens)

def get_prompt_numeric(question : str, background : str, choices : str) -> Tuple[str, str]:
    """
    Returns a tuple with a prompt and ideal response for GPT finetuning, specific to numeric questions
    """

    choices = literal_eval(choices)
    if isinstance(choices['max'], str):
        max_date = datetime.strptime(choices['max'], '%Y-%m-%d')
        min_date = datetime.strptime(choices['min'], '%Y-%m-%d')
        range = (max_date - min_date).days
        low = 0
        choices_text = '\n'.join(f'{token}: {min_date + ((i/num_split_len) * range)}' for i, token in enumerate(num_choice_tokens))
    else:
        range = choices['max'] - choices['min']
        low = choices['min']
        choices_text = '\n'.join(f'{token}: {low + ((i/num_split_len) * range)}' for i, token in enumerate(num_choice_tokens))
    
    prompt = f'Context: {background}\nQuestion: {question}\nWhich choice is closest?\n{choices_text}'

    return prompt


def get_prompt(question : str, background : str, choice_type : str, choices : str) -> str:
    


    try:
        choices = literal_eval(choices)
        assert 2 <= len(choices)
        choices_text = "\n".join(choices)
        

        prompt = f'Context: {background}\nQuestion: {question}\nChoices: {choices_text}'

        return prompt

    except Exception as e:
        print(f'warn on prompt {question} {background} {choices}')
        print(e)
        

        return None