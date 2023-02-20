print("Let's start")
import pandas as pd
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

oos.chdir(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 model development\structure")
test_set = pd.read_csv(r"csv\test_set_gpt2.csv")
test_set['True_end_abstract'] = test_set['abstract'].str.split().str[-20:].apply(' '.join)
test_set['Abstract'] = test_set['abstract'].str.split().str[:-20].apply(' '.join)

#dataset = Abstract(df['abstract'], truncate=False, gpt2_type="gpt2")
#Get the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
folder = input("Choose the folder [v1/v2]:\n>>>")
while True:
    if folder == "v1":
        last = os.listdir("pretrained_v1")[-1]
        model = GPT2LMHeadModel.from_pretrained(r"pretrained_v1\{last_}".format(last_=last))
        break
    elif folder == "v2":
        last = os.listdir("pretrained_v2")[-1]
        model = GPT2LMHeadModel.from_pretrained(r"pretrained_v2\{last_}".format(last_=last))
        break   
    else:
        folder = input("Choose the folder [v1/v2]:\n>>>")


def generate(
    model,
    tokenizer,
    prompt,
    entry_count=1,
    entry_length=30, #maximum number of words
    top_p=0.8,
    temperature=1.,
):
    model.eval()
    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in trange(entry_count):

            entry_finished = False
            if folder == "v1":
                generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
            elif folder == "v2":
                generated = torch.tensor(tokenizer.encode("<|startoftext|>"+prompt)).unsqueeze(0)
            
            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                #print(loss, logits)
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                #print(logits)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode("<|endoftext|>"):
                    entry_finished = True

                if entry_finished:

                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break
            
            if not entry_finished:
                output_list = list(generated.squeeze().numpy())
                output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 
                generated_list.append(output_text)
                
    return generated_list

#Function to generate multiple sentences. Test data should be a dataframe
def text_generation(test_data):
    generated_abstract = []
    for i in range(len(test_data)):
        x = generate(model.to("cpu"), tokenizer, test_data['Abstract'][i], entry_count=1, top_p=0.8, temperature=1., entry_length=100)
        generated_abstract.append(x)
    return generated_abstract


#Run the functions to generate the lyrics
generated_abstract = text_generation(test_set)


#Loop to keep only generated text and add it as a new column in the dataframe
my_generations=[]

for i in range(len(generated_abstract)):
    a = test_set['Abstract'][i].split()[-30:] #Get the matching string we want (30 words)
    b = ' '.join(a)
    c = ' '.join(generated_abstract[i]) #Get all that comes after the matching string
    my_generations.append(c.split(b)[-1])


test_set['Generated_abstract'] = my_generations


#Finish the sentences when there is a point, remove after that
final=[]

for i in range(len(test_set)):
    to_remove = test_set['Generated_abstract'][i].split('.')[-1]
    final.append(test_set['Generated_abstract'][i].replace(to_remove,''))

test_set['Generated_abstract'] = final


test_set.to_csv(r"csv\final_test_and_generated_{time}.csv".format(time = str(dt.datetime.now())[:10]+"-h"+str(dt.datetime.now())[11:13]+"-m"+str(dt.datetime.now())[14:16]))