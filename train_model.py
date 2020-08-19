import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

seed = 45

# data prep=====
df= pd.read_csv("wine_quality.csv")
y= df.pop("quality")

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size= 0.3, random_state= seed)

model1= RandomForestRegressor(max_depth=3, random_state=seed)
model1.fit(X_train, y_train)

train_score_rf= model1.score(X_train, y_train)*100
test_score_rf= model1.score(X_test, y_test)*100

with open("metrics.txt", "w") as outfile:
    outfile.write("Random Forest Training variance explained: %2.1f%% \n" % train_score_rf) 
    outfile.write("Random Forest Test variance explained: %2.1f%% \n" % test_score_rf)

model2= KernelRidge(alpha=1)
model2.fit(X_train, y_train)
train_score_kr= model2.score(X_train, y_train)*100
test_score_kr= model2.score (X_test, y_test)*100

with open("metrics.txt", "a") as outfile:
    outfile.write("Kernel Ridge Training variance explained: %2.1f%% \n" %train_score_kr) 
    outfile.write("Kernel Ridge Test variance explained: %2.1f%% \n" %test_score_kr)

# Feature Importance Plots =======

importance_rf = model1.feature_importances_
importance_rf
labels= df.columns

feature_df_rf =pd.DataFrame(list(zip( labels,importance_rf)), columns= ["feature", "importance"])
feature_df_rf= feature_df_rf.sort_values(by= "importance", ascending= False)
feature_df_rf


# image formatting
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

ax= sns.barplot(x= "importance",y="feature", data= feature_df_rf)
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
ax.set_title("Feature Importance for Random Forest")


plt.tight_layout()
plt.savefig("feature_importance_rf.png",dpi=120) 
plt.close()