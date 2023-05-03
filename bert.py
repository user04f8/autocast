import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from os.path import exists

model_name = "bert-base-uncased"
if exists(f'{model_name}_tokenizer') and exists(f'{model_name}_model'):
    with open(f'{model_name}_tokenizer', 'rb') as f:
        tokenizer = torch.load(f)
    with open(f'{model_name}_model', 'rb') as f:
        model = torch.load(f)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    with open(f'{model_name}_tokenizer', 'wb') as f:
        torch.save(tokenizer, f)
    with open(f'{model_name}_model', 'wb') as f:
        torch.save(model, f)

def encode_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def find_nearest_category(response, categories, model=model, tokenizer=tokenizer):
    response_vector = encode_text(response, model, tokenizer)
    category_vectors = [encode_text(cat, model, tokenizer) for cat in categories]

    similarities = [cosine_similarity(response_vector, cat_vector) for cat_vector in category_vectors]
    max_index = similarities.index(max(similarities))

    return categories[max_index]

if __name__ == "__main__":
    # sample usage

    response = "cat"
    categories = [
        "The text describes a dog.",
        "The text describes a cat.",
        "The text describes a goat.",
        "The text describes a horse."
    ]

    nearest_category = find_nearest_category(response, categories, model, tokenizer)
    print(f"The nearest category to the given response is: '{nearest_category}'.")
