#Using BLEU score to compare the real sentences with the generated ones
import statistics
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd


file =r"\final_test_and_generated_2023-01-25-h15-m03.csv"
file =r"\final_test_and_generated_2023-01-31-h15-m53.csv"
test_set = pd.read_csv(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 preparation\python training\csv"+file)

# print(test_set)
# print(test_set.shape[0])
scores=[]

for i in range(test_set.shape[0]):
    reference = test_set['True_end_abstract'][i]
    candidate = test_set['Generated_abstract'][i]
    try:
        scores.append(sentence_bleu(reference, candidate))
    except:
        continue
    #scores.append((reference, candidate))
print(scores)
print(statistics.mean(scores))