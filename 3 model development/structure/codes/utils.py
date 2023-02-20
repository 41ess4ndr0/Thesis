import torch
from tqdm import trange
import torch.nn.functional as F
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datetime as dt
import numpy as np
from joblib import load
import os
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import CountVectorizer



#Function for the model
def generate_(
    model,
    tokenizer,
    prompt,
    entry_count=10,
    entry_length=200, #maximum number of words
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
            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

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

                if next_token in tokenizer.encode("<|endoftext|>") and i>=50:
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

def get_csv(generated, value):
    df = pd.DataFrame()
    df["synthetic patent"] = generated
    df["xi_real_predicted"] = value
    # list_producion = np.array([generated,value]).reshape((1,2))

    
    df.to_csv(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 model development\structure\synthetic_patent_test\synthetic_patent{time}.csv".format(time = str(dt.datetime.now())[:10]+"-h"+str(dt.datetime.now())[11:13]+"-m"+str(dt.datetime.now())[14:16]))
    return df

def check_similarity(true_abstract,synthetic_abstract):
    """
    true_abstract -> list 1 value
    synthetic_abstract -> list 1+ values
    """
    if isinstance( true_abstract, list):
        true_ab = true_abstract[0]
    else:
        true_ab = true_abstract
    # print(true_ab)
    # print(synthetic_abstract)
    synthetic_abstract.insert(0,true_ab)
    all_ab = synthetic_abstract
    #print(all_ab, type(all_ab))
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings = model.encode(all_ab)
    cos = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])
    # print(cos, cos.mean())
    return  cos, cos.mean()

def correct_nouns (noun):
    noun = re.sub("'","",noun[1:-1],).split(", ")
    noun = list(set(noun))
    return ", ".join(noun)


def generate_test_2020(test):
    PATH_BIN = r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 model development\structure\bin for pipeline"
    PATH_TRAINING = r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 model development\structure"
    countvec = load(PATH_BIN+r"\CountVec.bin")
    scaler_X = load(PATH_BIN+r"\std_scaler_X.bin")
    scaler_y = load(PATH_BIN+r"\std_scaler_y.bin")
    pca =  load(PATH_BIN+r"\pca.bin")
    xgb = load(PATH_BIN+r"\model_XGBoost.bin")
    print("Loaded all packages")
    ### MODEL
    last = os.listdir(PATH_TRAINING + r"\pretrained_v2")[-1]
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(PATH_TRAINING + r"\pretrained_v2\{last_}".format(last_=last))

    print("Loaded the model")
    
    
    df_to_test = test.loc[:,("corrected_nouns", "xi_real", "abstract")]

    dict_generation = {"synthetic_patent":[], "predicted_xi_real":[] }
    print("NOUN ITERATION:")
    c = 1
    for nouns in df_to_test["corrected_nouns"]:
        
        initial_info = f"Write a patent with these nouns: {nouns.upper()}. "
        x = generate_(model.to("cpu"), tokenizer, initial_info, entry_count=1, top_p=0.7, temperature=1.4, entry_length= 200)

        #pipeline
        countvec2 = CountVectorizer(

                                analyzer='word', 
                                ngram_range=(1, 1), 
                                stop_words = "english"
        )

        print(x)
        #x.to_csc
        X2 = countvec2.fit_transform(x)
        X2 = pd.DataFrame(X2.toarray(), columns= countvec2.get_feature_names_out())

        fake_df = pd.DataFrame(columns= countvec.get_feature_names_out())
        X = fake_df.merge(X2, how="outer").loc[:,countvec.get_feature_names_out()].fillna(0)

        #X = countvec.transform(x)
        X = scaler_X.transform(X)
        X = pca.transform(X)

        prediction = xgb.predict(X)
        predicted_value = scaler_y.inverse_transform( prediction.reshape(-1,1))
        dict_generation["synthetic_patent"].append(x)
        dict_generation["predicted_xi_real"].append(predicted_value[0])
        print("##########")
        print(f"iteration n {c}")
        print("##########")
        c+=1

    df_to_test["synthetic_patent"] = dict_generation["synthetic_patent"]
    df_to_test["predicted_xi_real"] = dict_generation["predicted_xi_real"]

    print("correct assignment of generation")
    similarity= []
    true_abstract = df_to_test["abstract"]
    synthetic_abstract = df_to_test["synthetic_patent"]
    print("get true and synt")
    for i in range(len(df_to_test)):
        
        true = true_abstract[i]
        print(true)
        synthetic = synthetic_abstract[i]
        print(synthetic)
        similarity.append(check_similarity(true,synthetic)[1])
        print(f"iteration n {i}")
    df_to_test["similarity"] = similarity

    df_to_test.to_csv(PATH_TRAINING + r"\synthetic patent test 2020\test_2020_generation-{time}.csv".format(time = str(dt.datetime.now())[:10]+"-h"+str(dt.datetime.now())[11:13]+"-m"+str(dt.datetime.now())[14:16]))