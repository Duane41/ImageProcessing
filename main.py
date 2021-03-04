import numpy as np
import cv2
import datetime
import tkinter
from tkinter import filedialog 
import os
from multiprocessing import pool
import math

image_set = []

def get_input_image():
    '''
    DOCSTRING: Gets the input image and divides it into 400 20x20 parts
    INPUT: Image name
    OUTPUT: RGB vector
    '''
    tkinter.Tk().withdraw()

    #input_image = filedialog.askopenfilename(initialdir= os.getcwd(), title='Please select an image')
    input_image = "C:/Users/Duane de Villiers/Desktop/Hyperboliq/test_img.jpg"

    if ".jpg" not in input_image and ".png" not in input_image and ".jpeg" not in input_image:
        input_image = ""
        return print("Input file does not have the correct extension")
    else:
        input_image = cv2.imread(input_image, 1)

        img_vector = np.array(input_image)

        w,h,d = img_vector.shape
        #divides input image into 20x20 parts, i.e. 400 elements once flattened

        new_w = math.floor(h / 20)
        new_h = math.floor(w / 20)

        new_img = np.empty((20, 20, new_w, new_h))

        start_r = 0
        end_r = new_h - 1
        start_c = 0
        end_c = new_w - 1

        for r in range(1, 21):
            for c in range(1, 21):
                #img_vector[start_r:end_r][start_c:end_c]

                start_c = start_c + new_w
                end_c = new_w*(c+1) - 1
                print(start_c)
                print(end_c)
                print("-----------")

                
            start_c = 0
            end_c = new_w - 1

            start_r = start_r + new_h
            end_r = new_h*(r+1) - 1
        

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
            if ".jpg" not in image and ".png" not in image and ".jpeg" not in image:
                continue

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

def ToCIE(in_img_vector):
    '''
    DOCSTRING: Converts an image to the CIE image space
    INPUT: Average RGB vector of an image
    OUTPUT: CIE representation of the RGB image
    '''
    print(in_img_vector)
    var_X = in_img_vector[0] / 100.0
    var_Y = in_img_vector[1] / 100.0
    var_Z = in_img_vector[2] / 100.0

    print(var_X)
    if var_X > 0.008856:
         var_X = var_X ** ( 1/3 )
    else:
        var_X = ( 7.787 * var_X ) + ( 16 / 116 )

    if var_Y > 0.008856:
        var_Y = var_Y ** ( 1/3 )
    else:
        var_Y = ( 7.787 * var_Y ) + ( 16 / 116 )

    if var_Z > 0.008856:
        var_Z = var_Z ** ( 1/3 )
    else:
        var_Z = ( 7.787 * var_Z ) + ( 16 / 116 )

    return [( 116 * var_Y ) - 16, 500 * ( var_X - var_Y ), 200 * ( var_Y - var_Z )] 

def DeltaECIEDistance(input_img_1, input_img_2):
    '''
    DOCSTRING: Claculates the Delta E* CIE distance between two CIE-L*ab vectors
    INPUT: input_img_1 and input_img_2, two average RGB vectors of two images
    OUTPUT: The Delta E* CIE distance
    '''
    img_1 = ToCIE(input_img_1)
    img_2 = ToCIE(input_img_2)


    return math.sqrt(((img_1[0] - img_2[0]) ** 2 ) + ((img_1[1] - img_2[1]) ** 2) + ((img_1[2] - img_2[2]) ** 2 ))

get_input_image()