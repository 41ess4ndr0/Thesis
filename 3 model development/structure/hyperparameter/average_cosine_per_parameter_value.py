import pandas as pd
import os
os.chdir(r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 preparation\python training\hyperparameter")
files_temp = os.listdir("temperature")
files_top = os.listdir("top_p")
temp = {}
top_p = {}
for i in range(len(files_temp)):

    df_temp = pd.read_csv(r"temperature\{file}".format(file=files_temp[i])).drop("Unnamed: 0",axis=1)
    df_top_p = pd.read_csv(r"top_p\{file}".format(file=files_top[i])).drop("Unnamed: 0",axis=1)
    temp[i] = df_temp
    top_p[i] = df_top_p


first_temp = temp[0].set_index("temp_level")
first_top = top_p[0].set_index("top_p_level")

for i in range(1,len(files_temp)):
    first_temp = first_temp.merge(temp[i].set_index("temp_level"), left_index= True, right_index=True)
    first_top = first_top.merge(top_p[i].set_index("top_p_level"), left_index= True, right_index=True)

first_temp.columns = [f"cosine_file_{i}"for i in range(1,len(files_temp)+1)]
first_top.columns = [f"cosine_file_{i}"for i in range(1,len(files_temp)+1)]
print(first_top)
print(first_temp)
first_temp.to_csv(r"results\all_temp_cosines.csv")
first_top.to_csv(r"results\all_top_cosines.csv")


print("done")