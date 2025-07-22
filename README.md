# aquatic_insect_ml

<img align="left" width="25%" src="Fall_2024_Diptera_Hubbard_Brook.png">

The following repository contains code authored by the Bernhardt Lab's 2025 Data+/Climate+ team, including Abdulmlik Almuhanna, Uzair Chaudhry, Melosa Rao, and Boyu Tan, and uses a supervised machine learning model to identify insect from images of sticky traps (see image). 

Traps are deployed weekly along several streams in the Hubbard Brook Experimental Forest in New Hampshire during the snow-free season (approximately April through November) and, since 2018, have been counted by hand to identify dominant aquatic taxa (Diptera, Ephemeroptera, Plecoptera, Trichoptera). This record is maintained by Mike Vlah, Tamera Wooster, Chris Solomon, and Emily Bernhardt.

For visualizations of the existing insect record that has been manually counted so far, please visit: [hbwater.org](hbwater.org).

Please direct any questions or comments regarding this work to Heili Lowman at heili.lowman@duke.edu.


## Final Pipeline
`pipeline_final.ipynb` contains a script to detect and classify insects on a single image. The notebook contains running instructions for runnning this notebook on colab. This can also be adapted to run on clusters.
1. The user needs to upload image file in the same directory and update `org_img` variable to include image path.
2. The next code block loads models and requiremnts.
3. The thrid code block performes inference. A summary of results is saved in `class_summary.csv` and the each image path, predicted class and confidence is saved in `detailed_predictions.csv`. Individual cropped images in their corresponding prdiction folders are save in `results` folder.
5. The user can download everything to their machine by running forth code block.
6. The fifth code block resets the variables for new image inference

---
- The `YOLO-Segmentation (mAP0.6)` folder stores the results and the script to run yolo segmentation on our dataset. Note that for segmentation purposes all instances were labeled as insects and they will be classified using a seperate model

    - `yolo_segment_s.ipnyb` stores the script to run segmentation on the dataset with yolov8s model. This model acheived an mAP of rougly 0.6 on validation test and 0.58 on testing set

    - `segment` folder stores the results including metrics, graphs and model weights for training, validation and prediction (`train 4` stores results for training, `val` stores metrics for testing set with default conf and `val3` stores metrics for testing set with conf=0.15, `predict` stores 2 predicted inference images from testing set)

    - Note best weights for yolov8s segementation can be found in `YOLO-Segmentation (mAP0.6)/segment/train4/weights/best.pt`
- The `Resnet Classification Folder` stores the scripts and best weights for insect classication using the resnet model

    - The `Resnet_tensorflow_intial_version.ipynb` contains the script to augment images of all classes except dipteran to 400 images. We then performed used Resnet model for classification. This acheives 0.89 testing accuracy. Confusion matrix shows that model is good at predict most classes with 'other' being the sole exception (the model performs very poorely on the other class).
    - The `Resnet_Label_Smoothing_Confidence_Thresholding.ipynb` contains the script to  augment images of all classes except dipteran to 400 images. We one hot encode the labels and perform label smoothing by a factor of 0.1. The helps reduce model overconfidence about wrong prediction. Example: Class 2 -> one hot encoded: [0, 0, 1, 0, 0, 0] -> label smoothing [0.02, 0.02, 0.9, 0.02, 0.02]. We also visualised confidence of predicts and experimented with various confidence threshold.

