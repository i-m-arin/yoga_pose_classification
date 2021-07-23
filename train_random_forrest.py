import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import pickle
from opt import opt
args = opt


if __name__ == "__main__":

    # Loading the dataframe from the given path.
    input_path = args.inputpath
    df = pd.read_csv(os.path.join(input_path,"final_data.csv"))

    
    # Preparing the X and y values.
    y = df['y']
    X = df.loc[:,df.columns != 'y']

    
    # Splitting the total data into train and test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Initialising the RandomForest Classifier and fitting the train data.
    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)
    
    
    # Measuring the scores.
    train_score = clf.score(X_train,y_train)
    test_score = clf.score(X_test,y_test)

    
    # Saving the classifier as a pickle file.
    with open("random_forrest.pkl","wb") as f:
        pickle.dump(clf, f)
    
    print("Training Data Size: " + str(len(X_train)))
    print("Train Score: " + str(train_score))
    print("Cross-validation Data Size: " + str(len(X_test)))
    print("Cross Validation Score: " + str(test_score))

