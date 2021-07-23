import numpy as np
import pandas as pd
import os
import json
import cv2
from opt import opt
args = opt

if __name__ == "__main__":

    # Loading the json file from the given path.
    input_path = args.inputpath
    json_path = os.path.join(input_path,"keypoints.json")
    json_data = json.load(open(json_path))
    original_path = input_path.split('/')[-1]

    
    # For each entry in the json file, we are preparing the co-ordinates of the keypoints.
    image_paths = []
    points = []
    for data in json_data:
        image_id = data['image_id']
        path = os.path.join(original_path,image_id)
        img = cv2.imread(path)
        image_shape = img.shape
        keypoints = data['keypoints']
        image_paths.append(image_id)
        cur_points = []
        cur_points.append(tuple([image_shape[0],image_shape[1]]))
        for j in range(0,len(keypoints),3):
            cur_points.append(tuple([keypoints[j],keypoints[j+1]]))
        points.append(cur_points)

    point_names = [
        "image_shape","Nose","LEye","REye","LEar","REar","LShoulder",
        "RShoulder","LElbow","RElbow","LWrist","RWrist",
        "LHip","RHip","LKnee","RKnee","LAnkle","RAnkle"
    ]
    
    # Creating a dataframe with the co-ordinates.
    final_points = pd.DataFrame(points,columns=point_names)
    final_paths = pd.DataFrame(image_paths,columns = ["image_path"])
    final = pd.concat((final_paths,final_points),axis=1)
    
    # Loading the data.csv file to get the classes of each images.
    data_path = os.path.join(input_path,"data.csv")
    df = pd.read_csv(data_path)
    
    # Merging both the dataframes such that every image gets its corresponding class.
    last = pd.merge(final,df,on="image_path")
    output_path = os.path.join(args.outputpath,"keypoints.csv")
    
    # Saving the dataframe as csv file.
    last.to_csv(output_path,index=None)
