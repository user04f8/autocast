from typing import Tuple
from ast import literal_eval




def get_prompt(question : str, background : str, choice_type : str, choices : str) -> str:
    if choices == 'nan':
        # no choices == invalid as train
        return None


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