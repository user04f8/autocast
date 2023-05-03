import numpy as np
from ast import literal_eval

from config import numeric_exact_weighting, numeric_exact_weighting_dist_penalty
from gpt_utils import inv_numeric, num_choice_tokens

# format for t/f:
    # [#, #]
# format for mc:
    # [#, #, ..., #]
# format for num:
    # #
    # (all #s are from 0. to 1.)

def reformat_response(response : str, choices : str, qtype : str):
    """
    Reformats a response from GPT into the appropriate one-hot encoding or number,
    as required by the submission skeleton code.
    """
    if qtype == 'num':
        response_tokens = ' '.split(response)
        w = numeric_exact_weighting
        if len(response[0]) == 1 and 65 <= ord(response[0]) <= (65 + num_choice_tokens):
            coarse_num = inv_numeric(response[0])
        else:
            coarse_num = 0.5
            w = 1 # weight 100% the exact_num
        if len(response) == 4:
            try:
                exact_num = float(response[3])
            except:
                print('Warn:', f'invalid exact num {response[3]}')
        else:
            return coarse_num # weight 100% the coarse_num

        if (coarse_num - exact_num) > (1/num_choice_tokens):
            w = min(0, w - ((coarse_num - exact_num) * numeric_exact_weighting_dist_penalty))
        
        return (1-w) * coarse_num + w * exact_num
    else:
        choices = literal_eval(choices)
        if response.lower in choices ...
        return np.array()
