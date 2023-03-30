import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from tokenizer import preprocess_and_encode

#def preprocess_and_encode(_):
#    return _

input_sentence = 'Hello world'

class TransformerBinaryClassifier(nn.Module):
    def __init__(self, model_name):
        super(TransformerBinaryClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)

        

        self.output = nn.Linear(self.transformer.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]  # Get the [CLS] token representation
        logits = self.output(pooled_output)
        return nn.functional.softmax(logits, dim=1)

if __name__ == '__main__':
    # Example usage
    model_name = "distilbert-base-uncased"  # Use any other model you prefer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TransformerBinaryClassifier(model_name)

    # Use the input_sentence from the previous example
    inputs = tokenizer(preprocess_and_encode(input_sentence), return_tensors="pt")
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

    # Get the prediction
    prediction = model(input_ids, attention_mask)
    print(prediction)
