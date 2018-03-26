# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 09:46:23 2018

@author: CK
"""

#Load the Librarys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#loading the data with encode 
df_kick = pd.read_csv("C:/Users/red/Desktop/MachineLearning/KickStarter/ks-projects-201801.csv")

#%%

#knowning the main information of our data
print(df_kick.shape)
print(df_kick.info())

#%%
#Looking at the data
df_kick.describe()
df_kick.head(n=3)

#%%

percentual_sucess = round(df_kick["state"].value_counts() / len(df_kick["state"]) * 100,2)

print("State Percentual in %: ")
print(percentual_sucess)

plt.figure(figsize = (8,6))

ax1 = sns.countplot(x="state", data=df_kick)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=45)
ax1.set_title("Status Project Distribuition", fontsize=15)
ax1.set_xlabel("State Description", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)

plt.show()
#%%

main_cats = df_kick["main_category"].value_counts()
main_cats_failed = df_kick[df_kick["state"] == "failed"]["main_category"].value_counts()
main_cats_success = df_kick[df_kick["state"] == "successful"]["main_category"].value_counts()

#main_cats_failed and main_cats_success are pandas series objects

main_cats_success=main_cats_success.sort_index(ascending=True)
main_cats_failed=main_cats_failed.sort_index(ascending=True)

main_cats_combined = (main_cats_success + main_cats_failed)

print(main_cats_success)
print(main_cats_failed)
#%%
x1=main_cats_combined.index
sns.set_color_codes("pastel")
sns.barplot(y=main_cats_combined,x=x1,color="b",label="Total")

x2=main_cats_success.index
sns.set_color_codes("muted")
  
ax2 =sns.barplot(y=main_cats_success,x=x2,color="b",label="Success")
ax2.set_xticklabels(ax2.get_xticklabels(),rotation = 70)


plt.show()
#%%

#Normalization to understand the distribuition of the pledge
df_kick["pledge_log"] = np.log(df_kick["pledged"]+ 1)
df_kick["goal_log"] = np.log(df_kick["goal"]+ 1)

#df_fail_succ_susp = df_kick[df_kick["state"] == "failed"|]


#df_failed = df_failed.sort_index(ascending=True)
#df_success = df_success.sort_index(ascending=True)
#df_suspended = df_suspended.sort_index(ascending=True)

#%%
g = sns.FacetGrid(df_kick, hue="state", col="main_category", margin_titles=True)
g=g.map(plt.scatter, "goal_log","pledge_log",edgecolor="w").add_legend()
#g.set_xlim(0, 8)
#g.set_ylim(0, 8)
#g.grid(True)
#g.yscale('log')
#%% Remove the null rows #%%
g = sns.FacetGrid(df_kick, hue="state", col="main_category", margin_titles=True)
g=g.map(plt.hist, "goal_log",edgecolor="w",bins=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]).add_legend()


#%%

df_success = df_kick[df_kick["state"] == "successful"]
corr=df_kick.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr,square=True,annot=True)


# Not very informative; Ofcourse, there is a good correlation between number of backers and amount pledged
# Need to go by the categories and goal amounts in each category to predict

#%%
#Split into train and test data
#X_train, X_test, y_train, y_test = train_test_split(Xarr,yarr,
                                                   # test_size=0.05,
                                                   # random_state=RANDOM_STATE)
                                                   
                                                   
#%% Remove the null rows #%%
h = sns.FacetGrid(df_kick, hue="state", col="category", margin_titles=True)
h=h.map(plt.hist, "goal_log",edgecolor="w",bins=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]).add_legend()
#%% Success by country

country_wise = df_kick["country"].value_counts()
country_wise_failed = df_kick[df_kick["state"] == "failed"]["country"].value_counts()
country_wise_success = df_kick[df_kick["state"] == "successful"]["country"].value_counts()

#main_cats_failed and main_cats_success are pandas series objects

country_wise_success=country_wise_success.sort_index(ascending=True)
country_wise_failed=country_wise_failed.sort_index(ascending=True)

country_wise_combined = (country_wise_success + country_wise_failed)

print(country_wise_success)
print(country_wise_failed)

z1=country_wise_combined.index
sns.set_color_codes("pastel")
sns.barplot(y=country_wise_combined,x=z1,color="b",label="Total")

z2=country_wise_success.index
sns.set_color_codes("muted")
  
ax2 =sns.barplot(y=country_wise_success,x=z2,color="b",label="Success")
ax2.set_xticklabels(ax2.get_xticklabels(),rotation = 70)


plt.show()

# All the countries seem to have the same success to fail ratio 1:2

#%% Build a model; do random forests on categories ; certain goal amount ranges and certain categories
# seem o have a higher success rate than others
# One-hot encoding on categories?
# state = success or fail 1 or 0; ignore all others

