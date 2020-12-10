# EC523 Project: Stock Price Predict

CONTENTS OF THIS FILE
---------------------

 * Introduction
 * Required Modules
 * Installation
 * Files Description
 * Maintainers

INTRODUCTION
------------

Stock price prediction has always been a challenging task because of the volatility in the stock market. 
In this project, we will use a deep learning(DL)-based method to predict stock market prices for a certain period. The method consists of a dedicated architecture of a stacked multilayer Long Short-Term Memory(LSTM) with attention mechanism.


 * For a full description of the project, visit the presentation video:
   https://youtu.be/Gp-2gFkL2AY

 * For a detailed report, visit the final report of this project:
   https://docs.google.com/document/d/1fI3ijNHPgbyYqFMXy6S5-oZd1ZymNAla/edit
   
   
   
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
 * Overview: folder of project overview ipynb file
 * Parameter Tuning: folder of ipynb file for parameter selection
 
 Files
 * data_overview: data information, descriptions and plots
 * main: main file with Data class and Config class
 * model: model file with Net class, train, predict and evaluate methods
 * arima: ARIMA model for stocks prediction (for comparison)
 
USAGE
------------
  ```
 python main.py
   ```
   

MAINTAINERS
-----------
Current maintainers:
 * Haoqi Gu - haoqigu@bu.edu
 * Fengxu Tu - fengxutu@bu.edu
 * Junwei Li - jly8@bu.edu
 * Jingyi Li - ljy668@bu.edu
 


 
 
 
 
 
 
 
 
  

   
  
