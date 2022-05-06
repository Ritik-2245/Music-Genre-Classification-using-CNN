
# Music-Genre-Classification-using-CNN
[GTZAN DATASET](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) _we have used GTZAN DATASET to train our CNN Model mainly GENRES-ORIGINAL having 10 folders with 10 genres 100 '.wav' files for each genre._

## PYTHON MODULES USED
* Tensorfolw.Keras
* sklearn
* librosa

   

## MFCC 
 Mel Frequency Cepstral Coefficients (MFCC) is an internal audio representation format which is easy to work on. This is similar to JPG format for images. This task was simplified by Davis and Murmelstein in the 1980's when they introduced the world with Mel Frequency Cepstral Coefficients (MFCCs) as a feature which is now widely used for audio analysis. First things first what does MFCC stands for it is an acronym for Mel Frequency Cepstral Coefficients which are the coefficients that collectively mae up an MFC. MFC is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear Mel scale of frequency.
   

## CNN
 Convolution Neural Network (CNN) are particularly useful for spatial data analysis, image recognition, computer vision, natural language processing, signal processing and variety of other different purposes. They are biologically motivated by functioning of neurons in visual cortex to a visual stimulus. What makes CNN much more powerful compared to the other feedback forward networks for image recognition is the fact that they do not require as much human intervention and parameters as some of the other networks such as MLP do. This is primarily driven by the fact that CNNs have neurons arranged in three dimensions. CNNs make all of this magic happen by taking a set of input and passing it on to one or more of following main hidden layers in a network to generate an output.
* Convolution Layers
* Pooling Layers
* Fully Connected Layers
   

## Flowchart of our model
![flowchart](/screenshot/flowchart.jpg)

## CNN MODEL
![cnn model](/screenshot//cnn_model.jpg )


## Accuracy and error vs epochs graph for training data and validation data
  Here it has achieved the accuracy of 91% for training data set, 86% for validation data and around 88% for testing data set. 
![Ac_err_vs_epochs](/screenshot/acurracy_and_errorVSepochs.png)

## Conclusion
This proves that our effort gives a CNN based programmed music sort arrangement framework. Their element vectors are determined utilizing Mel-Spectrum. The python language built in librosa bundle assistances to remove highlights and, consequently it helps in giving great boundaries to the organization preparing. The learning exact nesses are demonstrated to be 91% for training ,88% for testing and for Mel. Subsequently, this approach is completely based upon user end, they can give their own custom songs as input and then finally they got the output as classified genres and promising for the arrangement of the gigantic information base of melodies into the individual type. The future work will zero in on building up the framework further to arrange the tunes dependent on the mindset. This will be useful in discovering which sort of music can lessen pressure in an individual while tuning in it. This is useful for musical treatment, which could be utilized for in concert a piece of specific music relying upon the individual's anxiety. This work should be additionally stretched out for such a framework.
