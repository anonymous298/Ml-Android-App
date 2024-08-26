from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]

        df = np.array([int_features])

        df = scaler.transform(df)

        prediction = model.predict(df)

        if prediction == 1:
            # return jsonify({'label':1})
            return render_template('index.html', prediction=prediction)
        
        else:
            # return jsonify({'label':0})
            return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)