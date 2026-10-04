from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Chargement du modèle entraîné
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Récupération des données saisies dans le formulaire
        data = {
            'age': [int(request.form['age'])],
            'sex': [request.form['sex']],
            'bmi': [float(request.form['bmi'])],
            'children': [int(request.form['children'])],
            'smoker': [request.form['smoker']],
            'region': [request.form['region']]
        }
        
        df_input = pd.DataFrame(data)
        prediction = model.predict(df_input)[0]
        
        prediction_text = f"Frais d'assurance estimés : ${prediction:,.2f}"
        
        return render_template('index.html', prediction_result=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)