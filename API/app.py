import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open('modelPickle.pkl', 'rb'))


@app.route("/")
def index():
    return render_template('index.html', pred=0)


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(request.form['Gender']),
            float(request.form['Age']),
            float(request.form['family_history_with_overweight']),
            float(request.form['FAVC']),
            float(request.form['FCVC']),
            float(request.form['NCP']),
            float(request.form['CAEC']),
            float(request.form['SMOKE']),
            float(request.form['CH2O']),
            float(request.form['SCC']),
            float(request.form['FAF']),
            float(request.form['TUE']),
            float(request.form['CALC']),
            float(request.form['MTRANS'])]
    data = [data]
    print(data)

    predictions = model.predict(data)
    print('INFO Predictions: {}'.format(predictions))

    # class_ = np.where(predictions == np.amax(predictions, axis=1))[1][0]

    dic_pred = {0: "Insufficient_Weight", 1: "Normal Weight", 2: "Overweight type I",
                3: "Overweight type II", 4: "Obesity type I", 5: "Obesity type II",
                6: "Obesity type III"}

    predi = dic_pred[predictions[0]]

    return render_template('index.html', pred=predi)


def main():
    """Run the app."""
    app.run(host='0.0.0.0', port=8000, debug=False)


if __name__ == '__main__':
    main()
