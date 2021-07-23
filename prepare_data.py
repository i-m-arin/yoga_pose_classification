import numpy as np
import os
import pandas as pd
from opt import opt
args = opt


if __name__ == "__main__":

    # Preparing the directories from the given path
    input_path = args.inputpath
    categories = os.listdir(input_path)
    categories.sort()

    # Preparing lists of image paths and their corresponding classes.
    cnt = 0
    total_image_path = []
    y = []
    allowed_formats = ["jpg","jpeg","png"]

    for category in categories:
        cur_path = os.path.join(input_path, category)
        images = os.listdir(cur_path)
        for image in images:
            if image.split('.')[-1] not in allowed_formats:
                continue
            cur_image = os.path.join(category,image)
            total_image_path.append(cur_image)
            y.append(cnt)
        cnt+=1

    total_image_path = np.array(total_image_path)
    y = np.array(y)
    total_image_path = np.reshape(total_image_path,(-1,1))
    y = np.reshape(y,(-1,1))

    train_data = np.concatenate((total_image_path,y),axis=1)
    
    
    # Creating of dataframe with image path and its corresponding class.
    df = pd.DataFrame(train_data,columns = ["image_path","y"])
    
    
    # Saving the dataframe as a csv file.
    output_path_csv = os.path.join(args.outputpath,"data.csv")
    df.to_csv(output_path_csv,index=None)

    
    # Saving all the image paths in a text file.
    output_path_text = os.path.join(args.outputpath,"text.txt")
    file = open(output_path_text,"w")

    for index,row in df.iterrows():
        
        file.write(row[0])
        file.write("\n")

    file.close()