import json
import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import T5EncoderModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class T5EncoderClassifier(nn.Module):
    def __init__(self, size, num_labels=2, strategy=0):
        super().__init__()
        
        if size=="base":
            in_features = 768
        elif size=="large":
            in_features = 1024
            
        self.tokenizer = T5Tokenizer.from_pretrained("t5-"+size)
        self.model = T5EncoderModel.from_pretrained("t5-"+size)
        self.classifier = nn.Linear(in_features, num_labels)
        self.strategy = strategy
        
    def forward(self, context, response):        
        max_len = 768
        data = [[x, y] for x, y in zip(context, response)]
        batch = self.tokenizer(data, max_length=max_len, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(input_ids=batch["input_ids"].cuda(), attention_mask=batch["attention_mask"].cuda())
        sequence_output = outputs["last_hidden_state"][:, 0, :]
        logits = self.classifier(sequence_output)        
        return logits
    
    def convert_to_probabilities(self, logits):
        if self.strategy == 0:
            probs = F.softmax(logits, 1)
        elif self.strategy == 1:
            probs = F.gumbel_softmax(logits, tau=1, hard=False)
        elif self.strategy == 2:           
            probs = F.gumbel_softmax(logits, tau=1, hard=True)
        return probs
    
    def output_from_logits(self, context, decoded_logits, response_mask):
        '''
        b: batch_size, l: length of sequence, v: vocabulary size, d: embedding dim
        decoded_probabilities -> (b, l, v)
        attention_mask -> (b, l)
        embedding_weights -> (v, d)
        output -> (b, num_labels)
        '''
        # encode context #
        max_len = 768
        batch = self.tokenizer(context, max_length=max_len, padding=True, truncation=True, return_tensors="pt")
        context_ids = batch["input_ids"].cuda()
        context_mask = batch["attention_mask"].cuda()
        context_embeddings = self.model.encoder.embed_tokens(context_ids)
        
        # encode response #
        decoded_probabilities = self.convert_to_probabilities(decoded_logits)
        embedding_weights = self.model.encoder.embed_tokens.weight
        response_embeddings = torch.einsum("blv, vd->bld", decoded_probabilities, embedding_weights)
        
        # concatenate #
        merged_embeddings = torch.cat([context_embeddings, response_embeddings], 1)
        merged_mask = torch.cat([context_mask, torch.tensor(response_mask).cuda()], 1)        
        outputs = self.model(inputs_embeds=merged_embeddings, attention_mask=merged_mask)
        sequence_output = outputs["last_hidden_state"][:, 0, :]
        logits = self.classifier(sequence_output)
        return logits
    
