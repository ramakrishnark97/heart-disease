from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model_hrt.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = request.form['AgeCategory']
    val2 = request.form['DiffWalking']
    val3 = request.form['Stroke']
    val4 = request.form['PhysicalHealth']
    val5 = request.form['Diabetic']
    arr = np.array([val1, val2, val3, val4,val5])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])

    return render_template('index.html', data=int(pred))


if __name__ == '__main__':
    app.run(debug=True)
