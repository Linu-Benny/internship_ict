from flask import Flask, render_template, request
import pickle
import numpy as np
import bz2

# Create Flask app
app = Flask(__name__)

# Function to decompress and load the pickle model
def decompress_pickle(file):
    with bz2.BZ2File(file, 'rb') as data:
        return pickle.load(data)

# Lazy loading the model
model = None

def load_model():
    global model
    if model is None:
        model = decompress_pickle('rf_model.pbz2')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    load_model()

    try:
        # Get user input
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
        else:
            raise ValueError("Invalid Credit Mix value")

        Outstanding_Debt = float(request.form['Outstanding_Debt'])
        Monthly_Balance = float(request.form['Monthly_Balance'])
        Total_Num_Accounts = float(request.form['Total_Num_Accounts'])
        Total_Monthly_Expenses = float(request.form['Total_Monthly_Expenses'])

        # Make prediction
        features = np.array([[Annual_Income, Interest_Rate, Num_of_Loan, Delay_from_due_date, Num_of_Delayed_Payment, Changed_Credit_Limit, Num_Credit_Inquiries, Credit_Mix, Outstanding_Debt, Monthly_Balance, Total_Num_Accounts, Total_Monthly_Expenses]])
        prediction = model.predict(features)

        if prediction[0] == 2:
            result = "Credit Score is Standard"
        elif prediction[0] == 1:
            result = "Credit Score is Good"
        else:
            result = "Credit Score is Poor"

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('prediction.html', pred_res=result)

if __name__ == "__main__":
    app.run(debug=True)



