import argparse
import cv2
import numpy as np
import os.path
import pandas as pd


coords = []
npix=[]
pixels_count_print=[]
drawing = False









'''

change imagepath to image file name, must be in the same folder as the python script!
change imagename to whatever you want to this image in the excel file


Region of interest crop1:

draw roi rectangle around whole brain, then slide until whole brain white
hit q button

Region of interest crop2:
draw roi over infarct, then slide until infarct size accurate
hit q


thats it!
change imagepath and imagename and move on



'''









imagepath='IMG_5033.JPG'
imagename='section 1B'
animal='mouse_1'


file_path=animal+'.csv'


#searched folder for file, to prevent accidental overwrite
if os.path.exists(file_path):
    df=pd.read_csv(file_path)
    
else:
    df=pd.DataFrame(columns=['Image','Name','Healthy','Infarcted'])

    

#redundant but keep in incase Arun wants to do cortical/subcortical
get=1






#intial function to crop and get brain size
def main():
    
    image = cv2.imread(imagepath)
    if image is not None:
        
        cv2.namedWindow('CapturedImage', cv2.WINDOW_NORMAL)
        cv2.imshow('CapturedImage', image)
        
        cv2.setMouseCallback('CapturedImage', click_and_crop, image)
        while True:
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                

                cv2.destroyAllWindows()
                break


#produce mouse drag crop effect
def click_and_crop(event, x, y, flag, image):
   
    #stolen from se, dont like global functions personally
    global coords, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        
        drawing = True
        
        coords = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        
        if drawing is True:
            
            clone = image.copy()
            cv2.rectangle(clone, coords[0], (x, y), (0, 255, 0), 2)
            cv2.imshow('CapturedImage', clone)
    elif event == cv2.EVENT_LBUTTONUP:
        
        drawing = False
        
        coords.append((x, y))
        if len(coords) == 2:
            
            ty, by, tx, bx = coords[0][1], coords[1][1], coords[0][0], coords[1][0]
            
            roi = image[ty:by, tx:bx]
            height, width = roi.shape[:2]
            if width > 0 and height > 0:
                
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)



                thresh=140

                blur = cv2.GaussianBlur(gray,(5,5),0)
                ret3,th3 = cv2.threshold(blur,thresh,255,cv2.THRESH_BINARY)
                im_bw_inv = cv2.bitwise_not(th3)

                contour, _ = cv2.findContours(im_bw_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contour:
                    cv2.drawContours(im_bw_inv, [cnt], 0, 255, -1)

                nt = cv2.bitwise_not(th3)
                im_bw_inv = cv2.bitwise_or(im_bw_inv, nt)



                def on_change(val):

                    imageCopy = gray.copy()
                    imageCopy_3 = gray.copy()

                    thresh=val/100

                    blur = cv2.GaussianBlur(gray,(5,5),0)
                    ret3,th3 = cv2.threshold(blur,thresh,255,cv2.THRESH_BINARY)
                    im_bw_inv = cv2.bitwise_not(th3)

                    contour, _ = cv2.findContours(im_bw_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contour:
                        cv2.drawContours(im_bw_inv, [cnt], 0, 255, -1)

                    nt = cv2.bitwise_not(th3)
                    im_bw_inv = cv2.bitwise_or(im_bw_inv, nt)

                    cv2.imshow(windowName, im_bw_inv)
                    
                    n_white_pix_2 = np.sum(im_bw_inv == 255)
                    npix.append(n_white_pix_2/171.1)




                windowName='Whole Brain Area'

                cv2.imshow(windowName, im_bw_inv)
                cv2.createTrackbar('slider', windowName, 10000, 25500, on_change)

                n_white_pix = np.sum(im_bw_inv == 255)
                #lazy append function as opposed to doing properly, also performs normalisation to mm
                npix.append(n_white_pix/171.1)





#intial function to crop and get infarct size, bigger function
def main_2():
    
    image = cv2.imread(imagepath)
    if image is not None:
        
        cv2.namedWindow('CapturedImage', cv2.WINDOW_NORMAL)
        cv2.imshow('CapturedImage', image)
        
        cv2.setMouseCallback('CapturedImage', click_and_crop_2, image)
        while True:
           
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
               
                cv2.destroyAllWindows()
                break



def click_and_crop_2(event, x, y, flag, image):
    
    global coords, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        
        drawing = True
        
        coords = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        
        if drawing is True:
            
            clone = image.copy()
            cv2.rectangle(clone, coords[0], (x, y), (0, 255, 0), 2)
            cv2.imshow('CapturedImage', clone)
    elif event == cv2.EVENT_LBUTTONUP:
        
        drawing = False
        
        coords.append((x, y))
        if len(coords) == 2:
            
            ty, by, tx, bx = coords[0][1], coords[1][1], coords[0][0], coords[1][0]
            
            roi = image[ty:by, tx:bx]
            height, width = roi.shape[:2]
            if width > 0 and height > 0:
                


                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)



                thresh=140

                blur = cv2.GaussianBlur(gray,(5,5),0)
                ret3,th3 = cv2.threshold(blur,thresh,255,cv2.THRESH_BINARY)
                im_bw_inv = cv2.bitwise_not(th3)

                contour, _ = cv2.findContours(im_bw_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contour:
                    cv2.drawContours(im_bw_inv, [cnt], 0, 255, -1)

                nt = cv2.bitwise_not(th3)
                im_bw_inv = cv2.bitwise_or(im_bw_inv, nt)



                def on_change(val):

                    imageCopy = gray.copy()
                    imageCopy_3 = gray.copy()

                    thresh=val/100

                    blur = cv2.GaussianBlur(gray,(5,5),0)
                    ret3,th3 = cv2.threshold(blur,thresh,255,cv2.THRESH_BINARY)
                    im_bw_inv = cv2.bitwise_not(th3)

                    contour, _ = cv2.findContours(im_bw_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contour:
                        cv2.drawContours(im_bw_inv, [cnt], 0, 255, -1)

                    nt = cv2.bitwise_not(th3)
                    im_bw_inv = cv2.bitwise_or(im_bw_inv, nt)

                    cv2.imshow(windowName, im_bw_inv)





                    imageCopy_2=roi.copy()

                    #shitty multiblur for smoothing, leading to canny edge detection
                    blur = cv2.GaussianBlur(im_bw_inv,(9,9),0)
                    blur = cv2.blur(im_bw_inv,(2,2))
                    kernel = np.ones((30,30),np.uint8)
                    
                    blur = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
                    
                    edges = cv2.Canny(blur,220,220)
                    kernel = np.ones((2,2),np.uint8)
                    #dilate edges to FILL GAPS
                    edges= cv2.dilate(edges,kernel,iterations = 1)
                    kernel = np.ones((5,5),np.uint8)
                    #prepare contours for infarct detection
                    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(im_bw_inv, contours, -1, (0, 255, 0), 1)

                    hierarchy = np.squeeze(hierarchy)

                    for i in range(len(contours)):
                    
                        color = (0, 0, 255)
                        color2 = (255, 255, 255)
                        
                        #prepare 
                        test2= cv2.drawContours(imageCopy_2, contours, i, color, -1)

                        #create black background
                        #drawing = np.zeros((imageCopy_2.shape[0], imageCopy_2.shape[1], 3), np.uint8)
                        pixels=cv2.drawContours(imageCopy_2, contours, i, color2, -1)
                        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
                        
                        #thresh to only white
                        thresh = 254

                        
                        pixels = cv2.threshold(pixels, thresh, 255, cv2.THRESH_BINARY)[1]


                        #way too jaggy= not biological, iterative smoothing with max kernel
                        pixels = cv2.blur(pixels,(9,9))
                        pixels = cv2.GaussianBlur(pixels,(45,45),0)
                        pixels = cv2.GaussianBlur(pixels,(45,45),0)
                        pixels = cv2.GaussianBlur(pixels,(45,45),0)
                        pixels = cv2.GaussianBlur(pixels,(45,45),0)
                        pixels = cv2.GaussianBlur(pixels,(45,45),0)

                        #thresh to include smoothed areas
                        thresh = 145

                        
                        pixels = cv2.threshold(pixels, thresh, 255, cv2.THRESH_BINARY)[1]

                        #prep overlay for display to Aruns window
                        background_2 = imageCopy_3
                        overlay_2 = pixels

                        added_image_2 = cv2.addWeighted(background_2,0.4,overlay_2,0.1,0)



                        pixels_count_2=cv2.countNonZero(pixels)
                        pixels_count_print.append(pixels_count_2/171.1)
                        

                    cv2.imshow(windowName2, added_image_2)




                windowName='Otsu Thresholding'

                cv2.imshow(windowName, im_bw_inv)
                cv2.createTrackbar('slider', windowName, 15000, 20000, on_change)


                placeholder=roi.copy()

                blur = cv2.GaussianBlur(im_bw_inv,(9,9),0)
                blur = cv2.blur(im_bw_inv,(2,2))
                kernel = np.ones((30,30),np.uint8)
                
                blur = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
                
                edges = cv2.Canny(blur,220,220)
                kernel = np.ones((2,2),np.uint8)
                edges= cv2.dilate(edges,kernel,iterations = 1)
                kernel = np.ones((5,5),np.uint8)
                contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


                hierarchy = np.squeeze(hierarchy)




                hull = []

                
                for i in range(len(contours)):
                    
                    hull.append(cv2.convexHull(contours[i], False))





                for i in range(len(contours)):
                    
                    color = (0, 0, 255)
                    color2 = (255, 255, 255)
                    

                    
                    test2= cv2.drawContours(placeholder, contours, i, color, -1)



                    drawing = np.zeros((placeholder.shape[0], placeholder.shape[1], 3), np.uint8)
                    pixels=cv2.drawContours(placeholder, contours, i, color2, -1)
                    pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)

                    thresh = 254

                    
                    pixels = cv2.threshold(pixels, thresh, 255, cv2.THRESH_BINARY)[1]


                    pixels = cv2.blur(pixels,(9,9))
                    pixels = cv2.GaussianBlur(pixels,(45,45),0)
                    pixels = cv2.GaussianBlur(pixels,(45,45),0)
                    pixels = cv2.GaussianBlur(pixels,(45,45),0)
                    pixels = cv2.GaussianBlur(pixels,(45,45),0)
                    pixels = cv2.GaussianBlur(pixels,(45,45),0)

                    thresh = 145

                   
                    pixels = cv2.threshold(pixels, thresh, 255, cv2.THRESH_BINARY)[1]



                    


                    pixels_count=cv2.countNonZero(pixels)
                    

                    background = gray
                    overlay = pixels

                    added_image = cv2.addWeighted(background,0.4,overlay,0.1,0)









                windowName2='Stroke Buster CannyEdge:Contour:MultiGauss'

                cv2.imshow(windowName2, added_image)
                cv2.createTrackbar('slider', windowName2, 15000, 20000, on_change)







                #lazy append function as opposed to doing properly, also performs normalisation to mm
                pixels_count_print.append(pixels_count/171.1)










#main launches brain area
#main_2 launches infarct area
main()
main_2()


print('filepath: ' + imagepath + ' healthy: ' +  str(npix[0]) + ' infarcted: ' + str(pixels_count_print[0])) 

slide=[imagepath,imagename,npix[-1],pixels_count_print[-1]]

df = df.append(pd.Series(slide, index=['Image','Name','Healthy','Infarcted']), ignore_index=True)
df= df.drop(df.columns[:-4], axis=1)

df.to_csv(file_path)

df
