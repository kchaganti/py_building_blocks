# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# create a dummy dataframe to practice

df_success = pd.DataFrame({'Category': pd.Categorical(['Music','Technology','Art','Film']),
                    'Num Success': pd.Series([40,80,30,90])})
df_failure = pd.DataFrame({'Category': pd.Categorical(['Art','Music','Technology','Film']),
                    'Num Failure': pd.Series([40,50,10,80])})

df_success=df_success.set_index('Category')
df_failure=df_failure.set_index('Category')

df_success=df_success.sort_index(ascending=True)
df_failure=df_failure.sort_index(ascending=True)

df_combined = (df_success['Num Success'] + df_failure['Num Failure'])
#df_percent_success = 100* df_success['Num Success']/ (df_success['Num Success'] + df_failure['Num Failure'])

#print(df_combined)
#print(df_percent_success)
print(df_success)
print(df_failure)
#%%
x1=df_combined.index
sns.set_color_codes("pastel")
sns.barplot(y=df_combined,x=x1,color="b",label="Total")

y2=df_success['Num Success']
x2=y2.index
sns.set_color_codes("muted")
sns.barplot(y=y2,x=x2,color="b",label="Success")

#sns.barplot(data=df_success)
plt.show()