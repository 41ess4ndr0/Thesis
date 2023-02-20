import pandas as pd
PATH=r"C:\Users\laudi\OneDrive\Desktop\Tesi_workspace\repository\3 preparation\python training\synthetic patent test 2020"
get = pd.read_csv(PATH+r"\test_2020_generation-2023-02-16-h17-m00.csv")

get.to_excel(PATH+r"\test_2020_generation-2023-02-16-h17-m00.xlsx")