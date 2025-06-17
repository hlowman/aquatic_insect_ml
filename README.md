# aquatic_insect_ml

<img align="left" width="33%" src="Fall_2024_Diptera_Hubbard_Brook.png">

The following repository contains code authored by the Bernhardt Lab's 2025 Data+/Climate+ team, including Abdulmlik Almuhanna, Uzair Chaudry, Melosa Rao, and Boyu Tan, and uses a supervised machine learning model to identify insect from images of sticky traps (see image). 

Traps are deployed weekly along several streams in the Hubbard Brook Experimental Forest in New Hampshire during the snow-free season (approximately April through November) and, since 2018, have been counted by hand to identify dominant aquatic taxa (Diptera, Ephemeroptera, Plecoptera, Trichoptera). This record is maintained by Mike Vlah, Tamera Wooster, Chris Solomon, and Emily Bernhardt.

For visualizations of the existing insect record that has been manually counted so far, please visit: [hbwater.org](hbwater.org).

Please direct any questions or comments regarding this work to Heili Lowman at heili.lowman@duke.edu.

- The `YOLO-Segmentation (mAP0.6)` folder stores the results and the script to run yolo segmentation on our dataset. Note that for segmentation purposes all instances were labeled as insects and they will be classified using a seperate model

    - `yolo_segment_s.ipnyb` stores the script to run segmentation on the dataset with yolov8s model. This model acheived an mAP of rougly 0.6 on validation test and 0.58 on testing set

    - `segment` folder stores the results including metrics, graphs and model weights for training, validation and prediction (`train 4` stores results for training, `val` stores metrics for validation set and `val3` stores metrics for testing set, `predict` stores 2 predicted inference images from testing set)

    - Note best weights for yolov8s segementation can be found in `YOLO-Segmentation (mAP0.6)/segment/train4/weights/best.pt`

