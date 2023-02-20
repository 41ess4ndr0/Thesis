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
from utils import generate_, check_similarity


os.chdir(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 model development\structure")
test_set = pd.read_csv(r"csv\test_set_gpt2.csv")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# folder = input("Choose the folder [v1/v2]:\n>>> ")
folder = "v2"
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
        folder = input("Choose the folder [v1/v2]:\n>>> ")


#function for generation
def text_generation(test_data, temperature_=1., top_p_=0.75):
    generated_abstract = []
    for i in range(len(test_data)):
        x = generate_(model.to("cpu"), tokenizer, test_data['Abstract'][i], entry_count=3, top_p=top_p_, temperature=temperature_)
        generated_abstract.append(x)
    return generated_abstract




def get_similarity(test_set, temperature=1., top_p=0.75):
    generated_abstract = text_generation(test_set, top_p_=top_p, temperature_=temperature)

    my_generations=[]

    for i in range(len(generated_abstract)):
        part_remove = test_set['Abstract'][i].split()[-20:] #Get the matching string we want (20 words)
        part_remove = ' '.join(part_remove)



        all_generation_together = ' '.join(generated_abstract[i]) #Get all that comes after the matching string
        
        sentences = all_generation_together.split("<|endoftext|>")

        coll = []
        for ele in sentences:
            if len(ele) > 15: #avoinding useless sentence 
                new = re.sub(part_remove,"",re.sub("|","",re.sub("|startoftext|","",re.sub("<|endoftext|>","",ele ))))
                coll.append(new)
        my_generations.append(coll)

    print(my_generations)
    test_set['Generated_abstract'] = my_generations
    cosine_mean = []
    for i in range(len(test_set)):
        sim = check_similarity(test_set['True_end_abstract'][i],test_set['Generated_abstract'][i])
        cosine_mean.append(sim[1])
    return cosine_mean

#test_set['Cosine_similarity'] = get_similarity(test_set)


parameters = {
    "TEMPERATURE"   :   np.linspace(0.2, 2, 10), #[0.2 0.4 0.6 0.8 1.  1.2 1.4 1.6 1.8 2. ]
    "TOP_P"         :   np.linspace(0.2, 1, 5)

}


def tunning(test_set, parameter, name_parameter):
    collection = {"TEMPERATURE":[],"TOP_P":[]}
    PATH = r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 model development\structure\hyperparameter"

    if name_parameter == "TEMPERATURE":
        for level in parameter:
            cosine_mean_row = get_similarity(test_set, temperature=level)
            print(cosine_mean_row, type(cosine_mean_row))
            collection["TEMPERATURE"].append((level,np.mean(cosine_mean_row))) #mean of results
            #[f"{level}"]= cosine_mean_row.mean()

        df = pd.DataFrame(collection["TEMPERATURE"], columns=["temp_level","cosine"])
        print(df)
        df.to_csv(PATH+r"\temperature\hyperparameter_T-{time}.csv".format(time = str(dt.datetime.now())[:10]+"-h"+str(dt.datetime.now())[11:13]+"-m"+str(dt.datetime.now())[14:16]))
        max_row = df[df["cosine"]==df["cosine"].max()] 
        print("#######################")
        print(f"Max cosine: {max_row.iloc[0,1]}\nBest temperature: {max_row.iloc[0,0]}")
        print("")
        print("#######################")
        return df[df["cosine"]==df["cosine"].max()] 


    elif name_parameter == "TOP_P":
        for level in parameter:
            cosine_mean_row = get_similarity(test_set, top_p=level)
            print(cosine_mean_row, type(cosine_mean_row))
            collection["TOP_P"].append((level,np.mean(cosine_mean_row)))
    
        df = pd.DataFrame(collection["TOP_P"], columns=["top_p_level","cosine"])
        df.to_csv(PATH+r"\top_p\hyperparameter_P-{time}.csv".format(time = str(dt.datetime.now())[:10]+"-h"+str(dt.datetime.now())[11:13]+"-m"+str(dt.datetime.now())[14:16]))
        max_row = df[df["cosine"]==df["cosine"].max()] 
        print("#######################")
        print(f"Max cosine: {max_row.iloc[0,1]}\nBest top_p: {max_row.iloc[0,0]}")
        print("")
        print("#######################")
        return df[df["cosine"]==df["cosine"].max()] 



test_set = test_set[:10]
name= "TOP_P"
print(dt.datetime.now())
tunning(test_set,parameters[name],name )
print(dt.datetime.now())






# pd.DataFrame(columns=["temp_level","cosine"])
# pd.DataFrame(columns=["top_p_level","cosine"])
    # max_mean_temp = max(list(collection["TEMPERATURE"].values()))
    # max_mean_top_p = max(list(collection["TOP_P"].values()))


# #Finish the sentences when there is a point, remove after that
# final=[]

# # for i in range(len(test_set)):
# #     to_remove = test_set['Generated_abstract'][i].split('.')[-1]
# #     final.append(test_set['Generated_abstract'][i].replace(to_remove,''))
# # print(final)
# test_set['Generated_abstract'] = final












# generated_abstract = text_generation(test_set)


# #Loop to keep only generated text and add it as a new column in the dataframe
# my_generations=[]

# for i in range(len(generated_abstract)):
#     a = test_set['Abstract'][i].split()[-30:] #Get the matching string we want (30 words)
#     b = ' '.join(a)
#     c = ' '.join(generated_abstract[i]) #Get all that comes after the matching string
#     my_generations.append(c.split(b)[-1])

# #test_set['Generated_abstract'] = my_generations

# ###### HYPERPARAMETER




