import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib


data = pd.read_csv(r"F:\HND\4th stage\ML\assement\diabetes.csv")


X = data.drop('Outcome', axis=1)
y = data['Outcome']


model = DecisionTreeClassifier()
model.fit(X, y)


model_filename = 'F:/HND/ML/decision_tree_model_diabetes.pkl'
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")


loaded_model = joblib.load(model_filename)
new_values = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]
predictions = loaded_model.predict(new_values)
print("Predictions for new values:", predictions)
