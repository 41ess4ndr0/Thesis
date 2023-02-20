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

if "python training" not in os.listdir():
    os.makedirs(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 model development\structure")


os.chdir("structure")
print(os.getcwd())
# ### Prepare data
# df = pd.read_csv(rep_path+"\\99 generated csv\\ch 2\\raw_all_component.csv ")
# df = pd.read_csv(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\99 generated csv\ch 2\raw_all_component.csv ")
# print(df.index)
# test_set = df.sample(n = 40, random_state=1998)



# df = df.loc[~df.index.isin(test_set.index)]

df_full = pd.read_csv(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\1 csv\patent_data_RAW.csv")
df_full["issue_date_year"] = df_full.issue_date.apply(lambda x: x[-4:])
sample = df_full.sample(50000,random_state=1998) #it was the training set used for instruction

df_full= df_full[~df_full.index.isin(sample.index)]

test_set_2020 = df_full[df_full["issue_date_year"]=="2020"] #filter only 2020

test_set_2020.to_csv(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\99 generated csv\ch 3\test_2020.csv")

# df_full

# train_set = pd.read_csv(r"csv\training_set_gpt2.csv")
# print(len(df))

# df = df[df['abstract'].apply(lambda x: len(x.split(' ')) < 350)]

# print(len(df))


#####  SPLITING DATAFRAME FOR 




