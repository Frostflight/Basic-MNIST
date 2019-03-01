import numpy as np
import keras
from PIL import Image
import time
import os
import PIL.ImageOps    

model = keras.models.load_model("MNIST_Model.h5")
clear = lambda: os.system('cls')

while (True):
    netIn = np.array(Image.open("P:\\Jason Dunn\\Neural Networking\\Version 2\\letter.png").convert('L').getdata()).reshape(1,28,28,1) / 255
    output = model.predict(netIn)
    print("0:",[0][0])
    print("1:",output[0][1])
    print("2:",output[0][2])
    print("3:",output[0][3])
    print("4:",output[0][4])
    print("5:",output[0][5])
    print("6:",output[0][6])
    print("7:",output[0][7])
    print("8:",output[0][8])
    print("9:",output[0][9])
    print()
    print("You drew a ",output.argmax(),".",sep='')
    time.sleep(0.2)
    clear()
    
