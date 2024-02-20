import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import pickle

dataset_path = 'C:/Users/JALIL/Downloads/Customer_churn.xlsx'


def train_and_save_model(dataset_path):
    # Load the dataset
    dataset = pd.read_excel(dataset_path)
    dataset = dataset[dataset['TotalCharges'].notna()]
    dataset = dataset.drop(columns='customerID')

    label_encoder = LabelEncoder()
    dataset['Churn'] = label_encoder.fit_transform(dataset['Churn'])

    # Split the data into training and testing sets
    X = dataset.drop('Churn', axis=1)
    y = dataset['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Scaling
    from sklearn.preprocessing import MinMaxScaler
    col_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = MinMaxScaler()
    X[col_to_scale] = scaler.fit_transform(X[col_to_scale])
    # Feature preprocessing using one-hot encoding for categorical features
    categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                            'PhoneService', 'MultipleLines', 'OnlineSecurity',
                            'OnlineBackup', 'DeviceProtection', 'TechSupport',
                            'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
                            'InternetService', 'Contract', 'PaymentMethod']

    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)

    # Train the XGBoost model
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_train_encoded, y_train)

    # Save the trained model and preprocessor as pickle files
    with open('trained_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    with open('preprocessor.pkl', 'wb') as file:
        pickle.dump(preprocessor, file)

    # Evaluate the model on the test set
    X_test_encoded = preprocessor.transform(X_test)
    y_pred = model.predict_proba(X_test_encoded)
    value_percentage = y_pred * 100
    formatted_percentages = np.array([["{:.2f}%".format(value) for value in row] for row in value_percentage])
    pred = model.predict(X_test_encoded)
    accuracy = accuracy_score(y_test, pred)
    return formatted_percentages


def load_model_and_predict(input_data):
    # Load the trained model and preprocessor
    with open('trained_model.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)

    # Create a DataFrame from the input data
    user_data = pd.DataFrame([input_data])

    # Feature preprocessing using the loaded preprocessor
    user_data_encoded = preprocessor.transform(user_data)

    # Make predictions using the trained model
    prediction = model.predict_proba(user_data_encoded)[0]
    value_percentage = prediction * 100
    churn_probability = "{:.2f}%".format(value_percentage[0])


    # Categorize risk based on thresholds
    low_threshold = 0.3
    medium_threshold = 0.6

    if prediction[0] <= low_threshold:
        risk_of_churn = "Low"
    elif low_threshold < prediction[0] <= medium_threshold:
        risk_of_churn = "Medium"
    elif prediction[0] > medium_threshold:
        risk_of_churn = "High"

    return churn_probability, risk_of_churn


# Train and save the XGBoost model
train_and_save_model(dataset_path)
