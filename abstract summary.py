import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import MBartForConditionalGeneration, MBartTokenizer
from torch import cuda
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import re
import datetime
import os

device = 'cuda' if cuda.is_available() else 'cpu'


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.summary = self.data.summary
        self.text = self.data.text

    def __len__(self):
        return len(self.summary)

    def __getitem__(self, index):
        ctext = str(self.text[index])
        ctext = ' '.join(ctext.split())

        text = str(self.summary[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]

        if _ % 1 == 0:
            print({"Training Loss": loss.item()})

        if _ % 500 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=70,
                num_beams=20,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            print('preds', preds)
            print('target', target)
            if _ % 100==0:
                print(f'Completed {_}')
            predictions.extend(preds)
            actuals.extend(target)
            
    return predictions, actuals


def clean_text(text):
    text = re.sub('[^а-яА-яё|0-9]', ' ', str(text))
    text = ' '.join(text.split())
    return text


def main():

    TRAIN_BATCH_SIZE = 2
    VALID_BATCH_SIZE = 2
    TRAIN_EPOCHS = 1
    VAL_EPOCHS = 1
    LEARNING_RATE = 1e-4
    SEED = 42
    MAX_LEN = 512
    SUMMARY_LEN = 150

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")

    df = pd.read_csv(r"data.csv")
    df = df[['summary', 'text']]
    df = df.dropna().reset_index(drop=True)
    df['text'] = df.apply(lambda x: clean_text(x['text']), axis=1)
    df = df.dropna().reset_index(drop=True)
    print(df.shape)
    df.text = 'summarize: ' + df.text
    print(df.head())
    
    train_size = 0.90
    train_dataset = df.sample(frac=train_size, random_state=SEED)
    val_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))


    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
    val_set = CustomDataset(val_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)


    train_params = {
        'batch_size': TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }

    val_params = {
        'batch_size': VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
    }

    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    t1 = datetime.datetime.now()
    print(t1)
    for epoch in range(TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    t2 = datetime.datetime.now()
    print(t2)
    print(str(t2 - t1))
    for epoch in range(VAL_EPOCHS):
        predictions, actuals = validate(tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
        final_df.to_csv('predictions.csv')
   
    saved_model_dir = "./saved_model_summary/"

    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
        
    model.save_pretrained(saved_model_dir)
    tokenizer.save_pretrained(saved_model_dir)

if __name__ == '__main__':
    main()
