import numpy as np
from flask import Flask ,request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Breast_Cancer_Detection.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    features = [x for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    
    if prediction[0] == 0:
        ans = 'Benign'
    else:
        ans = 'Malignant'
    return render_template('index.html', prediction_text='Result is {}'.format(ans))

if __name__ == "__main__":
    app.run(debug=True)