print("Let's start")
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import trange
import torch.nn.functional as F
import os
import pandas as pd
import datetime as dt


os.chdir(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 model development\structure")
last = os.listdir("pretrained_v2")[-1]
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(r"pretrained_v2\{last_}".format(last_=last))
# def generate_(
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

sentence = ""+input("Digit your sentence:\n>>> ")
# sentence = """
# A developing blade member for regulating a thickness of a layer 
# of a developer on a peripheral surface of a rotatable developing roller 
# enclosing a magnet roller, wherein a scraper for scraping the developer 
# toward longitudinally inside of the developing roller is provided at a longitudinal end of the developing roller, 
# """D
input_info = tokenizer.encode(sentence, return_tensors ='pt')
generation = model.generate(input_info, max_length=450, do_sample=True, temperature = 2 , top_k=30, top_p=0.7)
#print(generation)
print(type(generation))
output = tokenizer.decode(generation[0], skip_special_tokens = True)
print(output)
# x = generate_(model.to("cpu"), tokenizer, sentence, entry_count=3, top_p=0.6, temperature=0.5, entry_length= 100)
# print(x)
# print(type(x))
# with open(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 preparation\python training\syntetic_patent_test\prova.txt", "w") as f:
#     f.write(x)
#     f.close()
# def get_csv(generated):
#     list_producion = generated
#     series = pd.Series(list_producion, name="syntetic patent")
#     series.to_csv(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 preparation\python training\syntetic_patent_test\syntetic patente{time}.csv".format(time = str(dt.datetime.now())[:10]+"-h"+str(dt.datetime.now())[11:13]+"-m"+str(dt.datetime.now())[14:16]))
#     return series
# print(get_csv(x))