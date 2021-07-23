import numpy as np
import pandas as pd
import os
from opt import opt
args = opt


if __name__ == "__main__":

    # Loading the keypoints.csv file.
    input_path = args.inputpath
    output_path = args.outputpath
    df = pd.read_csv(os.path.join(input_path,"keypoints.csv"))

    
    # Creating a list os lists where each entry will be the feature vector of an image.
    # For each image we will itrate through each pair of points in the keypoints and store the difference between its x-coordinates, y-coordinates and slope.
    data = []
    for index,row in df.iterrows():
        cur = []
        for i in range(2,19):
            for j in range(2,19):
                if i == j:
                    continue
                
                shape = row[1][1:-1].split(',')
                tup_1 = row[i][1:-1].split(',') 
                tup_2 = row[j][1:-1].split(',')
                
                x_shape = float(shape[0])
                y_shape = float(shape[1])
                
                fir_x = float(tup_1[0])
                fir_y = float(tup_1[1])
                sec_x = float(tup_2[0])
                sec_y = float(tup_2[1])
                
                x_diff = (fir_x - sec_x)
                y_diff = (fir_y - sec_y)
                
                if y_diff == 0:
                    sole = 1e10
                else:
                    slope = (x_diff) / (y_diff)
                
                x_diff = x_diff / x_shape
                y_diff = y_diff / y_shape
                
                cur.append(x_diff)
                cur.append(y_diff)
                cur.append(slope)
        
        data.append(cur)


    # Concatinating the feature vectors with their corresponding classes.
    final_data = pd.DataFrame(data,index=None)
    y_val = df['y']
    final_data = pd.concat((final_data,y_val),axis=1)

    # Saving the final feature vector as a csv file.
    final_data.to_csv(os.path.join(output_path,"final_data.csv"),index=None)
        
