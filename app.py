
import numpy as np
from flask import Flask, request, render_template
import pickle
model=pickle.load(open('iris.pkl', 'rb'))
app = Flask(__name__)



@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data1=request.form['a']
    data2=request.form['b']
    data3=request.form['c']
    data4=request.form['d']
    arr=[[data1,data2,data3,data4]]
    pred=model.predict(arr)
    return render_template('after.html',data=pred)


if __name__ == "__main__":
    app.run(debug=True)