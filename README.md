# FRUITS-IDENTIFICATION-SYSTEM-WITH-MOBILENETV2
The system builds and develops a system capable of accurately recognizing and classifying popular fruits. With the advancement of deep learning and deep neural networks, fruit identification will bring widespread applications in practice.

Results obtained
Testing with the test set has a pretty good accuracy rate, reaching about 92% with the validation file and 97% with the train.
Images with high false recognition rates often fall into two classes: “cherry” and “strawberry”.
The images with high correct recognition rates are in the other 8-class classes.

Evaluation
The model is still limited, cannot identify the fruit too accurately but is not too overfitting.

Causes
Some species have similar colors and shapes 
The number of images in the dataset is small, only about ~230 images for 1 dataset to train 1 type of fruit 
Photo quality is poor, broken, photo colors are almost similar

Conclusion
The model has been trained and recognizes 10 types of fruit with a relatively high accuracy of 98%, although the amount of data will still be limited. The model using MobileNetV2 has good learning and implementation value if developed into a software. Application to identify many different types of fruit, but requires a lot of dataset.
