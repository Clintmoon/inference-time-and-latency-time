# inference-time-and-latency-time
This is the code for calculating the inference time and latency time of the method proposed in the paper "An Encrypted Traffic Classification Approach Based on Path Signature Features and LSTM." The file "model.pth" contains all the parameters of the model, "dataset.npy" refers to the dataset required for testing, "main.py" includes all the code needed to compute the inference time and latency time, and "requirements.txt" lists all the dependencies needed for the environment.

The "dataset.npy" contains a total of 342 samples, which are input into the model with a batch size of 32, resulting in 11 batches. We use the average inference time and latency time of these 11 batches as the inference time and latency time of our method.
