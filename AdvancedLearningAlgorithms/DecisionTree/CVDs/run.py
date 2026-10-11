import tensorflow as tf
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

RANDOM_STATE = 55  

df = pd.read_csv('./data/heart.csv')

df.head()

cat_variables = ['Sex',
                 'ChestPainType',
                 'RestingECG',
                 'ExerciseAngina',
                 'ST_Slope']

df = pd.get_dummies(data= df, prefix= cat_variables, columns= cat_variables)
df.head()

features = [x for x in df.columns if x not in 'HeartDisease']

X_train, X_val, y_train, y_val = train_test_split(df[features], df['HeartDisease'], train_size= 0.8, random_state= RANDOM_STATE)

decision_tree_model = DecisionTreeClassifier(max_depth= 4, min_samples_split= 50, random_state= RANDOM_STATE).fit(X_train, y_train)

decision_tree_model = RandomForestClassifier(n_estimators= 100, max_depth= 4, min_samples_split= 50, random_state= RANDOM_STATE).fit(X_train, y_train)

n = int(len(X_train)*0.8) ## Let's use 80% to train and 20% to eval

X_train_fit, X_train_eval, y_train_fit, y_train_eval = X_train[:n], X_train[n:], y_train[:n], y_train[n:]

xgb_model = XGBClassifier(n_estimators= 500, learning_rate= 0.1, verbosity= 1, random_state= RANDOM_STATE)
xgb_model.fit(X_train_fit, y_train_fit, eval_set = [(X_train_eval, y_train_eval)], early_stopping_rounds = 10)

print(f"Metrics train:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_val),y_val):.4f}")
