import torch
from transformers import AutoTokenizer, AutoModel

import os
from hedge._global import CACHE_ROOT
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_ROOT,"ChineseBaseNAACL")


def __batch_process_(model, tokenizer, device, sentences):

   
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128, return_token_type_ids=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
    return embeddings



def get_sentence_embeddings(sentences, model_name = 'bert-large-uncased', batch_size = 2000):
    
    
    # model_name = 'bert-large-uncased'
    model_name = 'bert-base-uncased'
            
            
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    embeddings = []
    for start in range(0, len(sentences), batch_size):
        end = min(start + batch_size, len(sentences))
        print(f"Processing batch {start} to {end} of {len(sentences)}")
        batch_sentences = sentences[start:end]
        batch_embeddings = __batch_process_(model, tokenizer, device, batch_sentences)
        
        if start == 0:
            embeddings = batch_embeddings
        else:
            embeddings = torch.cat((embeddings, batch_embeddings), dim=0)

    return embeddings

if __name__ == "__main__":
    sentences = ["This is a sample sentence.", "Here is another one."]
    
    
    #  roberta-large    bert-large-uncased
    model_name = "bert-large-uncased"  # 可以选择其他模型，如""
    embeddings = get_sentence_embeddings(sentences, model_name)
