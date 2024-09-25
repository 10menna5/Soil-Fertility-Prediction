import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

#
path = r'C:\Users\menna\OneDrive\Documents\NASAproject\dataset1.csv'
data = pd.read_csv(path)

#
labels = data[['Output']]
features = data.drop(['Output', 'K', 'EC', 'OC', 'Zn', 'Fe', 'Mn', 'B'], axis=1)

#
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

#  RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train.values.ravel())

#  Streamlit
st.title('Soil Fertility Prediction')

ph = st.slider('Soil pH Level', 0.0, 14.0, 7.0)
nitrogen = st.number_input('Nitrogen (N) Content in Soil', min_value=0.0)
phosphorus = st.number_input('Phosphorus (P) Content in Soil', min_value=0.0)
kabreet = st.number_input('kabreet (S) Content in Soil', min_value=0.0)
nahas = st.number_input('nahas (Cu) Content in Soil', min_value=0.0)

input_data = pd.DataFrame({
    'N': [nitrogen],
    'P': [phosphorus],
    'pH': [ph],
    'S': [kabreet],
    'Cu': [nahas]
})

# زر للتنبؤ
if st.button('Predict Fertility'):
    # إجراء التنبؤ
    prediction = rf_classifier.predict(input_data)[0]

    # عرض النتيجة
    if prediction == 0:
        st.warning('The soil is not fertile!')
    elif prediction == 1:
        st.success('The soil is moderately fertile!')
    elif prediction == 2:
        st.success('The soil is highly fertile!')

