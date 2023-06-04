from flask import Flask, jsonify, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])

def do_prediction():

    json = request.get_json()
    model = joblib.load('clf_model.pkl')
    vec = joblib.load('vec.pkl')

    df = pd.DataFrame(json, index=[0])
    df_x_scaled = vec.transform(df['provision'])
    y_pred = model.predict(df_x_scaled)

    label = np.argmax(y_pred)
    if label == 0:
        label = "[Amendments]"
    elif label == 1:
        label = "[Counterparts]"
    elif label == 2:
        label = "[Governing Laws]"
    elif label == 3:
        label = "[Government Regulations]"
    elif label == 4:
        label = "[Terminations]"
    elif label == 5:
        label = "[Trade Relations]"
    elif label == 6:
        label = "[Trading Activities]"
    elif label == 7:
        label = "[Valid Issuances]"
    elif label == 8:
        label = "[Waivers]"
    elif label == 9:
        label = "[Warranties]"
    else:
        label = "Error: no matching label"

    result = {"Predicted Label" : label}
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)