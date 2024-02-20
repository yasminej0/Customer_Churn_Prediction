from flask import Flask, render_template, request

import model
app = Flask(__name__,static_url_path='/static')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form.to_dict()

        # Make predictions
        prediction = model.load_model_and_predict(data)
        churn_probability,risk_of_churn = prediction
        print(churn_probability,risk_of_churn)
        return render_template('results.html', risk_of_churn=risk_of_churn, churn_probability=churn_probability)
    else:
        print('error')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    app.run()
