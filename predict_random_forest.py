import pickle
import os
from opt import opt
import pandas as pd
args = opt

if __name__ == "__main__":

    # Given input path.
    input_path = args.inputpath

    # Loading the model.
    with open("random_forrest.pkl", "rb") as f:
        clf = pickle.load(f)
    
    # Loading the dataframe.
    df = pd.read_csv(os.path.join(input_path,"final_data.csv"))

    # Preparing X and y.
    y = df['y']
    X = df.loc[:,df.columns != 'y']

    # Measuring score.
    score = clf.score(X,y)

    print("Data Size: " + str(len(X)))
    print("Score : " + str(score))
