import numpy as np
import cv2
import datetime
import tkinter
from tkinter import filedialog 
import os
from multiprocessing import pool

image_set = []

def get_input_image():
    '''
    DOCSTRING: Gets the input image and divides it into 400 20x20 parts
    INPUT: Image name
    OUTPUT: RGB vector
    '''
    tkinter.Tk().withdraw()

    input_image = filedialog.askopenfilename(initialdir= os.getcwd(), title='Please select an image')

    if ".jpg" not in input_image or ".png" not in input_image or ".jpeg" not in input_image:
        input_image = ""
        return print("Input file does not have the correct extension")
    else:
        input_image = cv2.imread(input_image, 1)

        img_vector = np.array(input_image)
        #divides input image into 20x20 parts, i.e. 400 elements once flattened

        print(img_vector)

        return input_image

def calc_avg_rgb(in_img_name = ''):
    '''
    DOCSTRING: Calculates the average RGB vector for a given image. Will bes used as a helper function in calc_avg_rgb_set()
    INPUT: Image name and the path to the image
    OUTPUT: RGB vector
    '''

    if in_img_name == '':
        return print("Input image must be specified")
    
    img = cv2.imread(in_img_name, 1)

    img_vector = np.array(img)
    
    w,h,d = img_vector.shape
    
    img_vector.shape = (w*h, d)
   
    return tuple(img_vector.mean(axis=0))

def calc_avg_rgb_set():
    '''
    DOCSTRING: Calculates the average vector for all the images in the set
    INPUT: File of the folder with the images in
    OUTPUT: Average Vector for images
    '''
    global image_set

    tkinter.Tk().withdraw() 

    image_set_path = filedialog.askdirectory(initialdir= os.getcwd(), title='Please the file directory')

    print("Process initiated")
    for folder in os.listdir(image_set_path):
        img_folder_path = image_set_path + "/" + folder
        print("Processing " + img_folder_path + "...")
        for image in os.listdir(img_folder_path):
            img_full_path = img_folder_path + "/" + image
            image_set.append([calc_avg_rgb(img_full_path), img_full_path])

    print("Complete!")
        
def display_image(in_img_name = ''):
    '''
    DOCSTRING: Displays the input image
    INPUT: Image name and the path to the image
    OUTPUT: none
    '''
    if in_img_name == '':   
        return print("Input image must be specified")

    img = cv2.imread(in_img_name, 1)

    cv2.imshow('image', img)

    user_input = cv2.waitKey(0)

    if user_input == 27:         # wait for ESC key to exit

        cv2.destroyAllWindows()
    elif user_input == ord('s'): # wait for 's' key to save and exit

        cv2.imwrite('output' + datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)") + '.png', img)
        cv2.destroyAllWindows()

calc_avg_rgb_set()