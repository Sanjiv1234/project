import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import collections
import pickle
import time
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import *
from gtts import gTTS
import os
# from flask.ext.session import Session


model = load_model("model/model19h5")
#model._make_predict_function()

model_t = ResNet50(weights = "imagenet",input_shape = (224,224,3))



model_res = Model(model_t.input,model_t.layers[-2].output)
#model_res._make_predict_function()




def preprocess(img):
    img = image.load_img(img,target_size = (224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis = 0)
    img = preprocess_input(img)
    return img
    



def encode_img(img):
    img = preprocess(img)
   
    # with session.as_default():
    #     with session.graph.as_default():
    feature_vec = model_res.predict(img)
    feature_vec = feature_vec.reshape(-1,)
    return feature_vec

    


# In[7]:


with open("./vocab/word_to_idx.pkl","rb") as wtoi:
    word_idx = pickle.load(wtoi)
with open("./vocab/idx_to_word.pkl","rb") as itow:
    idx_word = pickle.load(itow)


# In[8]:


#final model



# In[9]:


#predict the caption of the image
def predict_caption(photo):
    in_text = "<start>"
    for i in range(35):
        sequence = [word_idx[w] for w in in_text.split() if w in word_idx]
        sequence = pad_sequences([sequence],maxlen = 35,padding ='post')
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_word[ypred]
        in_text += (' '+word)
        
        if word == "<end>":
            break
    final_cap = in_text.split()[1:]
    final_cap = " ".join(final_cap)
    
            
    return final_cap



# In[16]:


def caption_the_img(image):
    
    encimg = encode_img(image).reshape((1,2048))
    c = predict_caption(encimg)
    return c

def text_to_audio(image):
    mytxt = caption_the_img(image)
    language = "en"
    myobj = gTTS(text = mytxt,lang = language,slow = False)
    myobj.save("welcome.mp3")
    os.system("start welcome.mp3")
    return "welcome.mp3"
    #os.system("start welcome.mp3")
    
# caption_the_img("test.jpg")




