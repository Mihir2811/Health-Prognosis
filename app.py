from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST','GET'])
def predict():
    # Get form data
    print(request.form)
    data = [float(x) for x in request.form.values()]
    #input_data for no its not a heart attack = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
    #input_data for yes it is a heart attack = (63,1,3,150,268,1,1,187,0,3,0,2,2)
    final= [np.array(data)]
    print(data)
    print(final)

    # Predict
    prediction = model.predict(final)

    if(prediction==0):
        return render_template('result.html', pred=prediction)
    else:
        return render_template('result2.html', pred=prediction)

if __name__ == '__main__':
    app.run(debug=True)
