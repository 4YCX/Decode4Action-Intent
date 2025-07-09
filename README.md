# Decoder-4-Action-Intent
## 1.Introduction
### Time
A summary of my summer course, July, 2025
### Content
The main content of the work is to decode (classify) the movement intent of the hand using the NinaPro dataset
### Language
All work on this is written in the Python 

## 2.Statement
Firstly，It is sincerely recommended to understand the specific content of the Ninapro dataset before reading further.

Second，The data currently used to train the model is limited to Exercise A in DB1(Somebody）

Third，The model showed excellent results in the early phase, but we only measured the effectiveness of the model in the Task-Driven case with a simple ACC and Loss, and we did not discuss in depth the suitability and scalability of the model we use below

Finally, please feel free to leave suggestions in the comment area, if you have any questions, please feel free to discuss them in the comment area, and if you have more specific proposals and very creative research prospects, please feel free to Email me

## 3.Code
### Version 1 LSTM
The classic LSTM model uses LTSM for long-time series understanding, and the essence is to classify time-series understanding.

### Version 2 CNN+LSTM
The number of convolution layers in CNN can be changed freely, and we have not studied the relationship between the complexity cost and the effect of CNN algorithms。Using CNN as feature extraction, the final effect of the model is slightly improved compared with LSTM only.

### Version 3 CNN+LSTM+Attention+Transformer_Encoder
This model is our last and most comprehensive model, and it has very good scalability. There are major changes compared to the previous ones. Specifically, it is embodied in:

1. Replace the LSTM layer with a Transformer Encoder, which adds complexity overhead but also enhances the global awareness of long time series.
   
2. Add channel attention mechanism , which can perceive the relationship between multiple channels of sEMG data. This should be fundamental to a significant improvement for the model to scale to complex datasets

3. Combine frequency-domain features of the data，The data converted by the Encoder is paralleled with the frequency domain feature sequence.
   Noting：The frequency-domain feature extraction can take different forms.

4. The model also adds wavelet denoising in the data processing stage, the denoising method can be changed, and this module can be migrated, but it  only exists in the last version of the code.

## 4.Note
### All work has been open-sourced, and no requirements have been provided. You can check it out by cloning the repository.
