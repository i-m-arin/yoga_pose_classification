In this project we are classifying images of people into their respective yoga pose's class.


Dataset link: https://drive.google.com/drive/folders/1n1SztpBYPqY-5f079dk8JD4d5b6X-z9T?usp=sharing


First of all, we will find all the positions of the keypoints like Nose, Shoulders, Elbow, Knee etc from images. For that, we have used AlphaPose github repo(https://github.com/Amanbhandula/AlphaPose).

Then we have processed the keypoints and created a feature vector using the difference between the x and y values and slope.

Finally, We have used two types of classifier to classify them into their corresponding classes:
1. Support Vector Machine with one-vs-one decision function.
2. RandomForest Classifier.


To try this out:

Clone this repo.

Download the dataset from the given link and save it as "Train" and "Test".

Download the models:
1. duc_se.pth (https://drive.google.com/file/d/1OPORTWB2cwd5YTVBX-NE8fsauZJWsrtW/view)
2. yolov3-spp.weights (https://drive.google.com/file/d/1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC/view)
Place them into ./models/sppe and ./models/yolo respectively.

Download all the dependencies from requirements.txt file.

Follow the commands.txt file for the following commands.



Data Set Sizes:

Training Data Set Size: 712

Cross-Validation Data Set Size: 352

Testing Data Set Size: 490


Accuracy with SVM:

Training Set Accuracy: 0.9634831460674157

Cross Validation Set Accuray: 0.8636363636363636

Test Set Accuracy: 0.9428571428571428


Accuracy with RandomForest Classifier:

Training Set Accuracy: 1.0

Cross Validation Set Accuray: 0.9232954545454546

Test Set Accuracy: 0.9530612244897959
