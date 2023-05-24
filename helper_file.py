import numpy as np
import pandas as pd

from datasets import load_dataset,Dataset,DatasetDict
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
import pandas as pd
import time
from torch.utils.data import DataLoader
import tqdm

class MyTaskSpecificCustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels ):
        super(MyTaskSpecificCustomModel, self).__init__()
        self.num_labels = num_labels
        
        self.model = model = AutoModel.from_pretrained(checkpoint, config = AutoConfig.from_pretrained(checkpoint, 
                                                                                                       output_attention = True, 
                                                                                                       output_hidden_state = True ) )
        # New Layer
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, num_labels)
        
    def forward(self, input_ids = None, attention_mask=None, labels = None ):
        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
        
        last_hidden_state = outputs[0]
        
        sequence_outputs = self.dropout(last_hidden_state)
        
        logits = self.classifier(sequence_outputs[:, 0, : ].view(-1, 768 ))
        
        loss = None
        loss = None
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            # loss_func = RMSELoss()
            # loss_func = nn.MSELoss()
            loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1))
            
            return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


target_names = ["happiness", "sadness", "fear","anger","surprise","disgust","approval","anticipation","realization","desire","shame","relief","love","neutral"]
labels_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

target_label = dict(zip(labels_list, target_names))

## Setting number of labels
num_label_new = 14
dropout_rate = 0.2


checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.model_max_len=512
tokenizer.add_special_tokens = False

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

PATH = r'C:\Users\deepe\Deploy-main-NLP\model-distil-bert-save-checkpoint.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load
model_task_specific = MyTaskSpecificCustomModel(checkpoint=checkpoint, num_labels=num_label_new ).to(device)
model_task_specific.load_state_dict(torch.load(PATH))
model_task_specific.eval()

def tokenize(batch):
  return tokenizer(batch["text"], truncation=True, max_length=512)

hyperparams = {
    "Dropout Rate":0.2,
    "Optimizer":"AdaFactor",
    "Epoch Runs":30,
    "Learning Rate":0.001,
    "Schedular Type":"Linear",
    "Batch Size":16
 }


#got the input as below
def input_output(file_input):
    start_time = time.time()
    # text_set = ["Hello tell me your identiy","I am looking good"]
    
    # if single_input:
    #     text_set = list(file_input)
    # else:
    text_set = list(pd.read_excel(file_input)['Text'])
    # print(len(text_set))
    # print(text_set)
    #will remain below as it is
    label = np.zeros(len(text_set)).astype(int)
    test_flask_input = Dataset.from_pandas(pd.DataFrame(
        {'text': text_set,
        'label': label
        }))

    # print(test_flask_input)

    test_flask_input = test_flask_input.map(tokenize, batched=True)
    test_flask_input.set_format('torch', columns=["input_ids", "attention_mask", "label"] )

    def dataloader_return(inp = test_flask_input):   #give dataset as 
        test_dataloader = DataLoader(
            test_flask_input, batch_size = 16, collate_fn = data_collator
        )
        return test_dataloader

    torch.cuda.empty_cache()
    with torch.no_grad():
        torch.cuda.empty_cache()

    # progress_bar_eval = tqdm(range(len(dataloader_return()) ))   #add test_flask_input

    model_task_specific.eval()
    predictions_n = []
    prediction_probab_n = []
    for batch in dataloader_return(test_flask_input):
        batch = { k: v.to(device) for k, v in batch.items() }
        # print(batch)
        with torch.no_grad():
            outputs = model_task_specific(**batch)
            
        logits = outputs.logits
        predictions = torch.argmax(logits, dim = -1 )
        prediction_probab = torch.max(logits, dim = -1 ).values

        predictions_n.extend(predictions)
        prediction_probab_n.extend(prediction_probab)
        
        # progress_bar_eval.update(1)

    predicted_items = [target_label[i.item()] for i in predictions_n]
    predicted_logits = [i.item() for i in prediction_probab_n]

    end_time = time.time()

    total_time = end_time - start_time

    # print(predicted_items)
    # print(predicted_logits)
    return dict(zip(text_set,predicted_items)),"{:.2f}".format(total_time),"Distil-Bert-Base-Uncased-Model",f"{hyperparams}"
