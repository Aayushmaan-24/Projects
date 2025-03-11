# importing necessary libraries

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report
import pickle

# cleaning the data by dropping unnecessary columns

def clean_data():
    df = pd.read_csv("/home/aayushmaan/PycharmProjects/BreastCancerPredictorApp/DATA/data.csv")
    data = df.drop(columns = ['Unnamed: 32','id'],axis =1)
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0}) # mapping M -> 1, B -> 0
    return data


# creating the model

def create_model(data):
    X = data.drop(columns = 'diagnosis', axis = 1) # X is everything but the result (input)
    Y = data['diagnosis'] # Y is the result (output)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X) # standardizing the inputs

    # splitting into train and test data
    X_train , X_test , y_train , y_test = train_test_split(X_std,Y,test_size=0.2,stratify=Y,random_state=42)

    # training the Logistic Regression Model
    model = LogisticRegression()
    model.fit(X_train,y_train)

    # testing the model
    y_preds = model.predict(X_test)
    accuracy = accuracy_score(y_test , y_preds)
    results = classification_report(y_test,y_preds)
    print(f"Accuracy : {accuracy:.4f}") # accuracy
    print(f"Classification Report : {results}") # entire report

    return model , scaler



def main():
    data = clean_data() # gets the cleaned dataset
    model , scaler = create_model(data) # using the cleaned dataset, gets the model and scaler

    with open('MODEL/model.pkl','wb') as f:
        pickle.dump(model , f)

    with open('MODEL/scaler.pkl','wb') as g:
        pickle.dump(scaler , g)




if __name__ == '__main__':
    main()