from flask import Flask, render_template, request
import pickle
import numpy as np
import bz2
import logging

# Create flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Decompress and load the model only when needed
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data

@app.route('/')
def home():
    logging.info("Rendering home page")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the model on-demand
        model = decompress_pickle('rf_model.pbz2')

        # Get User input
        Annual_Income = float(request.form['Annual_Income'])
        Interest_Rate = float(request.form['Interest_Rate'])
        Num_of_Loan = float(request.form['Num_of_Loan'])
        Delay_from_due_date = float(request.form['Delay_from_due_date'])
        Num_of_Delayed_Payment = float(request.form['Num_of_Delayed_Payment'])
        Changed_Credit_Limit = float(request.form['Changed_Credit_Limit'])
        Num_Credit_Inquiries = float(request.form['Num_Credit_Inquiries'])
        Credit_Mix = request.form['Credit_Mix'].title()

        if Credit_Mix == "Standard":
            Credit_Mix = 2
        elif Credit_Mix == 'Good':
            Credit_Mix = 1
        elif Credit_Mix == 'Bad':
            Credit_Mix = 0

        Outstanding_Debt = float(request.form['Outstanding_Debt'])
        Monthly_Balance = float(request.form['Monthly_Balance'])
        Total_Num_Accounts = float(request.form['Total_Num_Accounts'])
        Total_Monthly_Expenses = float(request.form['Total_Monthly_Expenses'])

        # Make prediction
        prediction = model.predict([[Annual_Income, Interest_Rate, Num_of_Loan, Delay_from_due_date, Num_of_Delayed_Payment, Changed_Credit_Limit, Num_Credit_Inquiries, Credit_Mix, Outstanding_Debt, Monthly_Balance, Total_Num_Accounts, Total_Monthly_Expenses]])

        if prediction[0] == 2:
            result = "Credit Score is Standard"
        elif prediction[0] == 1:
            result = "Credit Score is Good"
        else:
            result = "Credit Score is Poor"

        return render_template('index.html', pred_res=result)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return "Error during prediction", 500

if __name__ == "__main__":
    app.run(debug=True)

