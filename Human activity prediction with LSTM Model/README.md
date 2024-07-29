# Human-activity-recongnition-using-LSTM-network

Human Activity Recognition, or HAR for short, is the problem of predicting what a person is doing based on a trace of their movement using sensors.

## Objective: 
This problem is about the classifying sequences of accelerometer data recorded by smart phones into known well-defined movements.

The raw data is not available, a pre-processed version of the dataset was made available. The pre-processing steps included:

1. Pre-processing accelerometer and gyroscope using noise filters.
2. Splitting data into fixed windows of 2.56 seconds (128 data points) with 50% overlap.Splitting of accelerometer data into gravitational (total) and body motion components.

The dataset can be found at Kaggle website:https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

### About the dataset 

"The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data." (taken from above mentioned link)

