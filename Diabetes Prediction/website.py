from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import streamlit as st
import pandas as pd


diabetes_model = pickle.load(open('model.sav', 'rb'))


def main():
    st.title('Diabetes Prediction')

    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level (mg/dL)')
    BloodPressure = st.text_input('Blood Pressure value (mm Hg)')
    Insulin = st.text_input('Insulin Level (mu U/ml)')
    BMI = st.text_input('BMI value (kg/m^2) ')
    Age = st.text_input('Age of the Person')

    diagnosis = ''

    if st.button('Diabetes Test Result'):

        input_data = [[Pregnancies, Glucose, BloodPressure, Insulin, BMI, Age]]
        print(input_data)

        input_data_as_numpy_array = np.array(input_data)

        dataset = pd.read_csv('diabetes.csv')
        dataset_X = dataset.iloc[:, [0, 1, 2, 4, 5, 7]].values
        sc = MinMaxScaler(feature_range=(0, 1))
        dataset_scaled = sc.fit_transform(dataset_X)

        transformed_data = sc.transform(input_data_as_numpy_array)
        print(diabetes_model.predict(transformed_data))
        msg = ''
        diagnosis = diabetes_model.predict(transformed_data)
        if(diagnosis[0] == 0):
            msg = 'Not Diabetic'
        if(diagnosis[0] == 1):
            msg = 'Diabetic'

        st.success(msg)


if __name__ == '__main__':
    main()
