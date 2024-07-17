from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
labels=["bad quality","good quality"]
# Load the trained model
model_path = 'wine.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    output = prediction[0]
   
    return render_template('index.html', prediction_text='Predicted Quality: {}'.format(labels[output]))

if __name__ == "__main__":
    app.run(debug=True)
