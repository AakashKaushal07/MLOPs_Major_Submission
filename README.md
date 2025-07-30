MLOPS Major Submission by G24AI1020 ( Aakash Kaushal ).

In the repo, we trained a model, quantized that model and then predicted a few sample outputs.
The training and quantization is done via github actions, while the predictions are made in a docker file, while in turn are running on github infra.

The CI-CD Pipeline  performs 3 jobs
1. Test suite : It Runs pytest. Must pass before others execute.
2. train and quantize :  Trains model, runs quantization, uploads artifacts test suite.It would only runn if Test Suite steps is successful.
3. build and test container :  Builds Docker image, runs container (must execute predict.py successfully) train and quantize.It would only run if the traina and quantize step completes.

R-squared values for both unquantized and quantized models are as follows :\
##### Trained Model
Model: LinearRegression
Loss (RMSE): 0.736
R^2 Score: 0.591
#### De Quantized Model
R^2 score with de-quantized parameters: 0.5834

#### File Sizes
model.joblib : 697B
quant_params.joblib : 418B
unquant_params.joblib : 414B

The file sizes are on my local system 
