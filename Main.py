from flask import Flask,jsonify,request
from flasgger import Swagger
from sklearn.externals import joblib
from flask_cors import CORS

import numpy as np
import pandas as pd

app = Flask(__name__)
Swagger(app)
CORS(app)




@app.route('/input/task', methods=['POST'])
def predict():
    """
        Ini Adalah Endpoint Untuk Memprediksi IRIS
        ---
        tags:
            - Rest Controller
        parameters:
          - name: body
            in: body
            required: true
            schema:
              id: Petal
              required:
                - numcol
                - yieldpercol
                - totalprod
                - stocks
                - priceperlb
                - year
              properties:
                numcol:
                  type: int
                  description: Please input with valid numcol.
                  default: 0
                yieldpercol:
                  type: int
                  description: Please input with valid yieldpercol.
                  default: 0
                totalprod:
                  type: int
                  description: Please input with valid totalprod.
                  default: 0
                stocks:
                  type: int
                  description: Please input with valid stocks.
                  default: 0
                priceperlb:
                  type: int
                  description: Please input with valid priceperlb.
                  default: 0
                year:
                  type: int
                  description: Please input with valid year.
                  default: 0
        responses:
            200:
                description: Success Input
        """
    new_task = request.get_json()

    numcol = new_task['numcol']
    yieldpercol = new_task['yieldpercol']
    totalprod = new_task['totalprod']
    stocks = new_task['stocks']
    priceperlb = new_task['priceperlb']
    year = new_task['year']

    X_New = np.array([[numcol,yieldpercol,totalprod,stocks,priceperlb,year]])


    clf = joblib.load('honeyRegressor.pkl')

    resultPredict = clf[0].predict(X_New)

    return jsonify({'message': format(resultPredict)})

if __name__ == '__main__':
    app.run(debug=True)


