from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)


model_path = os.path.abspath(r"F:\HND\ML\decision_tree_model_diabetes.pkl")


if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file not found: {model_path}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        bloodpressure = float(request.form['bloodpressure'])
        skinthickness = float(request.form['skinthickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])

       
        data = [[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]]
        prediction = model.predict(data)

        result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
