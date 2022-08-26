import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

fish_set = pd.read_csv('Fish.csv')

x = fish_set[["Weight","Length1","Length2","Length3","Height","Width"]]
y = fish_set["Species"]

label = LabelEncoder()
y = label.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=58)

sv = SVC(kernel = "linear").fit(x_train,y_train)

pickle.dump(sv,open('model.pkl','wb'))