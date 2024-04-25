#%%
# regression
import pandas as pd
from sklearn.linear_model import LinearRegression

# %%
data = pd.read_csv("../data/data1.csv")

# %%
X = data[["介入","性別","年齢"]]
Y = data["売り上げ"]

model = LinearRegression()
model.fit(X, Y)
# %%
coef = model.coef_
intercept = model.intercept_

print(coef)
print(intercept)


# %%
#ipw
from sklearn.linear_model import LogisticRegression

X = data[["介入","性別","年齢"]]
Y = data["売り上げ"]

def calc_propensity_scores(X, Z):
    model = LogisticRegression().fit(X,Z)
    propensity_score = model.predict_proba(X)
    return propensity_score

Z = X["介入"]
X = X.drop(columns = "介入")
propensity_score = calc_propensity_scores(X, Z)

ate_i = Y/propensity_score[:,1]*Z - Y/propensity_score[:,0]*(1-Z)
ate = ate_i.sum()/len(Y)
print(ate)




# %%
df_propensity_score = pd.DataFrame(propensity_score)
df_propensity_score.columns = ["介入なし", "介入あり"]
df_propensity_score_compare = pd.concat([data, df_propensity_score], axis = 1)
data_plot = df_propensity_score_compare

import seaborn as sns
sns.violinplot(x = "介入", y = "介入あり", data = data_plot, palette = "muted")



# %%
# %%

