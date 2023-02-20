import pandas as pd
import re
from utils import generate_test_2020
import random

def correct_nouns (noun):
    noun = re.sub("'","",noun[1:-1],).split(", ")
    random.seed(1998)
    noun = random.sample(list(set(noun)) ,5) 
    return ", ".join(noun)

test_2020 = pd.read_csv(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\\3 model development\structure\csv\training_set_gpt2.csv")
test_2020 = test_2020.iloc[30000:32000, :]#not launched as training for Tokenziation problem
test_2020 = test_2020[test_2020["issue_date_year"]==2020]
test_2020 = test_2020.sample(10, random_state=12).reset_index(drop=True)
test_2020["corrected_nouns"] = test_2020.noun.apply(correct_nouns)
# print(test_2020.corrected_nouns)
# print(test_2020.xi_real)
# print(test_2020.columns)

generate_test_2020(test_2020)
# test_2020["correct_nouns"] = test_2020.nouns.apply(correct_nouns)

# test_2020.to_csv(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\99 generated csv\ch 3\test_2020.csv")
