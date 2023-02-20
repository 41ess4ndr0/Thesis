import os
import pandas as pd
from joblib import load
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils import generate_ , get_csv

from sklearn.feature_extraction.text import CountVectorizer


#retriver all the packages
###BIN
PATH_BIN = r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 model development\structure\bin for pipeline"
PATH_TRAINING = r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 model development\structure"
countvec = load(PATH_BIN+r"\CountVec.bin")
scaler_X = load(PATH_BIN+r"\std_scaler_X.bin")
scaler_y = load(PATH_BIN+r"\std_scaler_y.bin")
pca =  load(PATH_BIN+r"\pca.bin")
rf = load(PATH_BIN+r"\model_RandFor.bin")
### MODEL
last = os.listdir(PATH_TRAINING + r"\pretrained_v2")[-1]
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(PATH_TRAINING + r"\pretrained_v2\{last_}".format(last_=last))



print("Uploaded all necessary packages")


print("let's start")

#initial_info = input("Formulate your sentence about the patent:\n>>> ")
initial_info = "Patent about WATER."
x = generate_(model.to("cpu"), tokenizer, initial_info, entry_count=1, top_p=0.8, temperature=1.3, entry_length= 300)

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

prediction = rf.predict(X)
predicted_value = scaler_y.inverse_transform( prediction.reshape(-1,1))

print(f"The predicted xi_real is: {predicted_value}")

get_csv(x, predicted_value)