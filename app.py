import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Feature columns used by the churn classification model.
BASE_FEATURE_COLUMNS = [
    'CreditScore',
    'Gender',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary',
]


def rebuild_churn_scaler(label_encoder_gender, onehot_encoder_geo):
    """Rebuild the classification scaler if scaler.pkl was overwritten by another pipeline."""
    data = pd.read_csv('Churn_Modelling.csv')
    data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    data['Gender'] = label_encoder_gender.transform(data['Gender'])

    geo_encoded = onehot_encoder_geo.transform(data[['Geography']]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography']),
        index=data.index,
    )

    data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)

    X = data.drop('Exited', axis=1)
    y = data['Exited']
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    churn_scaler = StandardScaler()
    churn_scaler.fit(X_train)
    return churn_scaler


def ensure_compatible_scaler(model, scaler, label_encoder_gender, onehot_encoder_geo):
    model_input_dim = int(model.input_shape[-1])
    scaler_feature_count = getattr(scaler, 'n_features_in_', None)
    scaler_features = list(getattr(scaler, 'feature_names_in_', []))

    has_regression_schema = 'Exited' in scaler_features and 'EstimatedSalary' not in scaler_features
    count_mismatch = scaler_feature_count is not None and scaler_feature_count != model_input_dim

    if has_regression_schema or count_mismatch:
        st.warning(
            'Detected incompatible scaler artifact. Rebuilding a churn-compatible scaler from the dataset.'
        )
        return rebuild_churn_scaler(label_encoder_gender, onehot_encoder_geo)

    return scaler


# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

scaler = ensure_compatible_scaler(model, scaler, label_encoder_gender, onehot_encoder_geo)

## streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Align columns with the scaler schema to prevent feature-name mismatch errors.
if hasattr(scaler, 'feature_names_in_'):
    expected_columns = list(scaler.feature_names_in_)
else:
    expected_columns = BASE_FEATURE_COLUMNS + list(onehot_encoder_geo.get_feature_names_out(['Geography']))

for column in expected_columns:
    if column not in input_data.columns:
        input_data[column] = 0.0

input_data = input_data[expected_columns]

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
