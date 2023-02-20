print("Let's start")
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import trange
import torch.nn.functional as F
import os
import pandas as pd
import datetime as dt
import re




os.chdir(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 model development\structure")
last = os.listdir("pretrained_v2")[-1]
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(r"pretrained_v2\{last_}".format(last_=last))


test_set = pd.read_csv(r"csv\test_set_gpt2.csv")
# test_set['True_end_abstract'] = test_set['abstract'].str.split().str[-20:].apply(' '.join)
# test_set['Abstract'] = test_set['abstract'].str.split().str[:-20].apply(' '.join)


def correct_nouns (noun):
    noun = re.sub("'","",noun[1:-1],).split(", ")
    noun = list(set(noun))
    return ", ".join(noun)

# test_set["corrected_nouns"] = test_set["noun"].apply(correct_nouns)
# print(test_set["corrected_nouns"][0] )
# test_set.to_csv(r"csv\test_set_gpt2.csv")

#sentence = ""+input("Digit your sentence:\n>>> ")
# sentence = """
# A developing blade member for regulating a thickness of a layer 
# of a developer on a peripheral surface of a rotatable developing roller 
# enclosing a magnet roller, wherein a scraper for scraping the developer 
# toward longitudinally inside of the developing roller is provided at a longitudinal end of the developing roller, 
# """D
# print(test_set.noun[0])
# print(type(test_set.noun[0]))
def generate_syntetic_abstract(noun_abstract):
    sentence = f"Create a patent with these nouns: {noun_abstract}.\n"
    input_info = tokenizer.encode(sentence, return_tensors ='pt')
    generation = model.generate(input_info, max_length=450, do_sample=True, temperature = 2. , top_k=30, top_p=0.6)
    #print(generation)
    print(type(generation))
    output = tokenizer.decode(generation[0], skip_special_tokens = True)
    return output

test_set["Syntetic_abstract"] = test_set["corrected_nouns"].apply(generate_syntetic_abstract)
test_set.to_csv(r"Syntetic abstract test from 8\generation_from_nouns_{date}.csv".format(date=str(dt.datetime.now())[:10]+"-h"+str(dt.datetime.now())[11:13]+"-m"+str(dt.datetime.now())[14:16]))