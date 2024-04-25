#%%
import pandas as pd
import matplotlib.pyplot as plt


# %%
data = pd.read_csv("../data/data2.csv")
data
# %%
intervention0 = data[data["補講フラグ"] == 0]
x0 = intervention0["前回テスト点数"]
y0 = intervention0["今回テスト点数"]

intervention1 = data[data["補講フラグ"] == 1]
x1 = intervention1["前回テスト点数"]
y1 = intervention1["今回テスト点数"]

plt.scatter(x0, y0, c = "red" ,label="Z=0")
plt.scatter(x1, y1, c = "blue" ,label="Z=1")
plt.show()
# %%
import pandas as pd
from sklearn.linear_model import LinearRegression

data
# %%
X = data[["前回テスト点数","補講フラグ"]]
Y = data["今回テスト点数"]

model = LinearRegression()
model.fit(X,Y)
model.coef_
# %%
