# EC523 Project: Stock Price Predict

CONTENTS OF THIS FILE
---------------------

 * Introduction
 * Required Modules
 * Installation
 * Files Description
 * Report
 * Maintainers

INTRODUCTION
------------

Stock price prediction has always been a challenging task because of the volatility in the stock market. 
In this project, we will use a deep learning(DL)-based method to predict stock market prices for a certain period. The method consists of a dedicated architecture of a stacked multilayer Long Short-Term Memory(LSTM) with attention mechanism.


 * For a full description of the project, visit the presentation video:
   https://youtu.be/Gp-2gFkL2AY

 * For a detailed report, visit the final report of this project:
   https://github.com/H40Q1/523_stock_predict/blob/main/Stock%20Price%20Predict.pdf
   
   
   
REQUIREMENTS MODULES
------------

This module requires the following modules:

 * Pandas (https://pandas.pydata.org/)
 * Numpy (https://numpy.org/)
 * Matplotlib (https://matplotlib.org/)
 * PyTorch (https://pytorch.org/)
 * Torchnlp (https://pytorchnlp.readthedocs.io/en/latest/)
 * Statsmodels (https://www.statsmodels.org/stable/index.html)
 * Sklearn (https://scikit-learn.org/stable/)
 
 
 
INSTALLATION
------------
Install as you would normally install modules
 * Pandas
 ```
 $ pip install pandas
   ```
 * Numpy 
  ```
 $ pip install numpy
   ```
 * Matplotlib: install from source (recommended)
  ```
$ python -m pip install .
   ```
 * PyTorch 
  ```
 $ pip install torch
   ```
 * Torchnlp 
  ```
 $ pip install pytorch-nlp
   ```
 * Statsmodels
  ```
 $ pip install statsmodels
   ```
 * Sklearn
  ```
 $ pip install -U scikit-learn
   ```

 
 
 
FILES DESCRIPTION
------------
 Folders
 * Data: folder of dataset
 * Figures: folder of project plots
 * Report: folder of project final report and presentation slides
 * Parameter Tuning: folder of ipynb file for parameter selection
 
 Files
 * data_overview: data information, descriptions and plots
 * main: main file with Data class and Config class
 * model: model file with Net class, train, predict and evaluate methods
 * arima: ARIMA model for stocks prediction (for comparison)
 * args: read parameter arguments from command line
 
USAGE
------------

 ```
python main.py [--hidden-size HIDDEN_SIZE]
               [--step TIME_STEP] [--layer LSTM_LAYERS]
               [--learning-rate LEARNING_RATE]
               [--epochs EPOCHS] [--batch-size BATCH_SIZE] 
               [--stock-path STOCK_PATH]
               [--stock-name STOCK_NAME]
               
```

REPORT
-----------
See [`Report/Stock Price Predict.pdf`](./Report)
   

MAINTAINERS
-----------
Current maintainers:
 * Haoqi Gu - haoqigu@bu.edu
 * Fengxu Tu - fengxutu@bu.edu
 * Junwei Li - jly8@bu.edu
 * Jingyi Li - ljy668@bu.edu
 


 
 
 
 
 
 
 
 
  

   
  
