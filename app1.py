import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Jeśli model nie istnieje, trenuj go od nowa
try:
    model = pickle.load(open("model.sv", 'rb'))
except FileNotFoundError:
    # Załaduj dane
    data = pd.read_csv("DSP_1.csv")
    data["Age"].fillna(data["Age"].mean(), inplace=True)
    data.dropna(subset=['Embarked'], inplace=True)

    # Przygotuj dane do trenowania
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    data["Sex"] = encoder.fit_transform(data["Sex"])
    data["Embarked"] = encoder.fit_transform(data["Embarked"])

    y = data["Survived"]
    X = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Trenowanie modelu
    model = RandomForestClassifier(n_estimators=20, random_state=0)
    model.fit(X_train, y_train)

    # Zapisz model
    with open("model.sv", 'wb') as file:
        pickle.dump(model, file)

