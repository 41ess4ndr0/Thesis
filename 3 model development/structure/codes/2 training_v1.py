print("Let's start")
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv
import re
import os
import datetime as dt


# PATH_FIND = re.compile(".*repository.")
# rep_path = re.search(PATH_FIND,os.getcwd()).group()
os.chdir(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 model development")

if "structure" not in os.listdir():
    os.makedirs(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 model development\structure")


os.chdir("structure")
print(os.getcwd())
# ### Prepare data
# df = pd.read_csv(rep_path+"\\99 generated csv\\ch 2\\raw_all_component.csv ")
df = pd.read_csv(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\99 generated csv\ch 2\raw_all_component.csv ")

##Create a very small test set to compare generated text with the reality
isfirst = input("Is it the first iteration? [Y/N]\n>>> ")
while True:
    if isfirst == "Y":
        test_set = df.sample(n = 40, random_state=1998)
        df = df.loc[~df.index.isin(test_set.index)]
        test_set = test_set.reset_index()
        df = df.reset_index()
        if "csv" not in os.listdir():
            os.makedirs("csv")
        test_set.to_csv(r"csv\test_set_gpt2.csv", index = False)
        df.to_csv(r"csv\training_set_gpt2.csv", index = False)
        break
    elif isfirst == "N":
        test_set = pd.read_csv(r"csv\test_set_gpt2.csv")
        df = pd.read_csv(r"csv\training_set_gpt2.csv")
        break
    else:
        isfirst = input("Is it the first iteration? [Y/N]\n>>> ")

## Choose your range
range_training = input("select your range[1,2,3,4,5,6,7,8,9,10]:\n>>>")
if range_training == "1":
    df = df.iloc[:1000, :] 
elif range_training == "2":
    df = df.iloc[1000:2000, :]
elif range_training == "3":
    df = df.iloc[2000:3000, :]
elif range_training == "4":
    df = df.iloc[3000:4000, :]
elif range_training == "5":
    df = df.iloc[4000:5000, :]
elif range_training == "6":
    df = df.iloc[5000:6000, :]
elif range_training == "7":
    df = df.iloc[6000:7000, :]
elif range_training == "8":
    df = df.iloc[7000:8000, :]
elif range_training == "9":
    df = df.iloc[8000:9000, :]

#For the test set only, keep last 20 words in a new column, then remove them from original column
test_set['True_end_abstract'] = test_set['abstract'].str.split().str[-20:].apply(' '.join)
test_set['Abstract'] = test_set['abstract'].str.split().str[:-20].apply(' '.join)

#### Useful class for Abstract

class Abstract(Dataset):
    def __init__(self, control_code, truncate=False, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.abstract = []
        c = 0 
        for row in df['abstract']:
            self.abstract.append(torch.tensor(
                self.tokenizer.encode(f"<|{control_code}|>{row[:max_length]}<|endoftext|>")
            ))      
            if c%2000 == 0:
                print(f"iter n {c}")
            c+=1
        if truncate:
            self.abstract = self.abstract[:2000] #it was 20000
        self.abstract_count = len(self.abstract)
        
    def __len__(self):
        return self.abstract_count

    def __getitem__(self, item):
        return self.abstract[item]


isfirst = input("Is it the first istruction for the model? [Y/N]\n>>> ")
while True:
    if isfirst == "Y":
        dataset = Abstract(df['abstract'], truncate=False, gpt2_type="gpt2")
        #Get the tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        break
    elif isfirst == "N":
        last = os.listdir("pretrained")[-1]
        dataset = Abstract(df['abstract'], truncate=False, gpt2_type="gpt2")
        #Get the tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained(r"pretrained\{last_}".format(last_=last))
        break
    else:
        isfirst = input("Is it the first model? [Y/N]\n>>> ")
#get the class
 


#Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None


# training function
def train(
    dataset, model, tokenizer,
    batch_size=16, epochs=5, lr=2e-5,
    max_seq_len=400, warmup_steps=200,
    gpt2_type="gpt2", output_dir=".", output_prefix="wreckgar",
    test_mode=False,save_model_on_epoch=False,save_model = False
):
    acc_steps = 100 #maybe useless
    device=torch.device("cuda")
    model = model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup( #https://huggingface.co/transformers/v3.0.2/main_classes/optimizer_schedules.html
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss=0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in tqdm(enumerate(train_dataloader)):
            
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    if save_model:
        model.save_pretrained(f"pretrained_v1//pretrained-{str(dt.datetime.now())[:10]}-h{str(dt.datetime.now())[11:13]}-m{str(dt.datetime.now())[14:16]}")
    return model


# ## instruction the model
model = train(dataset, model, tokenizer, batch_size=16, epochs=20, save_model_on_epoch = False, save_model=True, lr=2e-3)

# ## generation function
# def generate(
#     model,
#     tokenizer,
#     prompt,
#     entry_count=10,
#     entry_length=30, #maximum number of words
#     top_p=0.8,
#     temperature=1.,
# ):
#     model.eval()
#     generated_num = 0
#     generated_list = []

#     filter_value = -float("Inf")

#     with torch.no_grad():

#         for entry_idx in trange(entry_count):

#             entry_finished = False
#             generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

#             for i in range(entry_length):
#                 outputs = model(generated, labels=generated)
#                 loss, logits = outputs[:2]
#                 logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

#                 sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#                 cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

#                 sorted_indices_to_remove = cumulative_probs > top_p
#                 sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
#                     ..., :-1
#                 ].clone()
#                 sorted_indices_to_remove[..., 0] = 0

#                 indices_to_remove = sorted_indices[sorted_indices_to_remove]
#                 logits[:, indices_to_remove] = filter_value

#                 next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
#                 generated = torch.cat((generated, next_token), dim=1)

#                 if next_token in tokenizer.encode("<|endoftext|>"):
#                     entry_finished = True

#                 if entry_finished:

#                     generated_num = generated_num + 1

#                     output_list = list(generated.squeeze().numpy())
#                     output_text = tokenizer.decode(output_list)
#                     generated_list.append(output_text)
#                     break
            
#             if not entry_finished:
#                 output_list = list(generated.squeeze().numpy())
#                 output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 
#                 generated_list.append(output_text)
                
#     return generated_list

# #Function to generate multiple sentences. Test data should be a dataframe
# def text_generation(test_data):
#     generated_abstract = []
#     for i in range(len(test_data)):
#         x = generate(model.to("cpu"), tokenizer, test_data['Abstract'][i], entry_count=2)
#         generated_abstract.append(x)
#     return generated_abstract


# #Run the functions to generate the lyrics
# generated_abstract = text_generation(test_set)


# #Loop to keep only generated text and add it as a new column in the dataframe
# my_generations=[]

# for i in range(len(generated_abstract)):
#     a = test_set['Abstract'][i].split()[-30:] #Get the matching string we want (30 words)
#     b = ' '.join(a)
#     c = ' '.join(generated_abstract[i]) #Get all that comes after the matching string
#     my_generations.append(c.split(b)[-1])

# test_set['Generated_abstract'] = my_generations


# #Finish the sentences when there is a point, remove after that
# final=[]

# for i in range(len(test_set)):
#     to_remove = test_set['Generated_abstract'][i].split('.')[-1]
#     final.append(test_set['Generated_abstract'][i].replace(to_remove,''))

# test_set['Generated_abstract'] = final

# test_set.to_csv("final_test_and_generated.csv")

#Using BLEU score to compare the real sentences with the generated ones
# import statistics
# from nltk.translate.bleu_score import sentence_bleu

# scores=[]

# for i in range(len(test_set)):
#     reference = test_set['True_end_abstract'][i]
#     candidate = test_set['Generated_abstract'][i]
#     scores.append(sentence_bleu(reference, candidate))

# print(statistics.mean(scores))