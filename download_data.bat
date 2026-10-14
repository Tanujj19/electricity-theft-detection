@echo off
set KAGGLE_CONFIG_DIR=C:\Users\HP\.kaggle
C:\Users\HP\AppData\Local\Programs\Python\Python310\Scripts\kaggle.exe datasets download -d bensalem14/sgcc-dataset --unzip -p data\raw
