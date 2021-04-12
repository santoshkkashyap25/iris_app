

from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle

d=load_iris()
df=pd.DataFrame(d.data)
df["target"]=d.target
x = df.drop("target",axis="columns")
y = df["target"]

model = LogisticRegression()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=10)
model.fit(x_train,y_train)
pickle.dump(model,open('iris.pkl','wb')) 