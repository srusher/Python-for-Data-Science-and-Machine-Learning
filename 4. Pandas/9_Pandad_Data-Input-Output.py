import pandas as pd
import os


# Reading CSV, Excel, HTML, and SQL files

os.chdir(r"C:\Users\sjrus\Desktop\Machine Learning\4. Pandas")

csv = pd.read_csv('example')
print(csv)

# create a dataframe from csv file:
df = pd.read_csv('example')
df.to_csv('My_Output',index=False)


# reading different sheet in excel
print(pd.read_excel('Excel_Sample.xlsx', sheet_name='Sheet1'))

# writing to an excel file
df.to_excel('Excel_Sample2.xlsx',sheet_name='NewSheet')


# Reading HTML files
#data = pd.read_html('https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/')
#print(data[0].head())


# Reading SQL files
from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')

df.to_sql('my_table',engine)
sqldf = pd.read_sql('my_table',con=engine)
print(sqldf)