## CORONASIMULATE  Simulate coronagraph and Gerchberg-Saxton algorithm
#
# A simulation of a coronagraph and the Gerchberg-Saxton algorithm, in the
# context of NASA's Roman Space Telescope, developed to help teach ENCMP
# 100 Computer Programming for Engineers at the University of Alberta. The
# program saves output figures to PNG files for subsequent processing.
#
# Copyright (c) 2022, University of Alberta
# Electrical and Computer Engineering
# All rights reserved.
#
# Student name: Christian Madrona
# Student CCID: madrona
# Others: 
#         Nick Anthony Miras                                                               (6.0%)
#         Me                                                                               (91% )
#         https://stackoverflow.com/questions/48700162/drawing-and-shading-circle-sections (3%  )
# To avoid plagiarism, list the names of persons, Version 0 author(s)
# excluded, whose code, words, ideas, or data you used. To avoid
# cheating, list the names of persons, excluding the ENCMP 100 lab
# instructor and TAs, who gave you compositional assistance.
#
# After each name, including your own name, enter in parentheses an
# estimate of the person's contributions in percent. Without these
# numbers, adding to 100%, follow-up questions will be asked.
#
# For anonymous sources, enter pseudonyms in uppercase, e.g., SAURON,
# followed by percentages as above. Email a link to or a copy of the
# source to the lab instructor before the assignment is due.
#
import matplotlib.pyplot as plt
import numpy as np

def main():
    im = loadImage('300_26a_big-vlt-s.jpg')
    (im,Dphi,mask) = opticalSystem(im,300)
    images, errors = gerchbergSaxton(im,10,Dphi,mask)
    saveFrames(images, errors)

def loadImage(name):
    im = plt.imread(name)/255
    if len(im.shape) > 2:
        im = (im[:,:,0]+im[:,:,1]+im[:,:,2])/3
    im[im < 0] = 0
    im[im > 1] = 1
    return im


def opticalSystem(im,width):
    im, mask = occultCircle(im,width)
    (IMa,IMp) = dft2(im)
    rng = np.random.default_rng(12345)
    imR = rng.random(im.shape)
    (_,Dphi) = dft2(imR)
    im = idft2(IMa,IMp-Dphi)
    return (im,Dphi,mask)

# occultCircle() Function:
# accepts two arguments: im (array) and width (int)
# creates an black circle of specified width at the center of the image
# by turning all RGB of the affected rows and columns into zero
# returns the resulting image
def occultCircle(im,width):
    imS = im.copy()
    mask = np.full(im.shape,False)
    xcenter = imS.shape[1] / 2
    ycenter = imS.shape[0] / 2
    radius = width / 2

    for y in range(0, imS.shape[0]+1):
        for x in range(0, imS.shape[1]+1):
            check = ((x-xcenter)**2) + ((y-ycenter)**2) # checks if the xy 
            if check <= radius**2:
                imS[y,x] = 0
                mask[y,x] = True
    return imS, mask

# (IMa,IMp) = dft2(im) returns the amplitude, IMa, and phase, IMp, of the
# 2D discrete Fourier transform of a grayscale image, im. The image, a 2D
# array, must have entries between 0 and 1. The phase is in radians.
def dft2(im):
    IM = np.fft.rfft2(im)
    IMa = np.abs(IM)
    IMp = np.angle(IM)
    return (IMa,IMp)

# im = idft2(IMa,IMp) returns a grayscale image, im, with entries between
# 0 and 1 that is the inverse 2D discrete Fourier transform (DFT) of a 2D
# DFT specified by its amplitude, IMa, and phase, IMp, in radians.
def idft2(IMa,IMp):
    IM = IMa*(np.cos(IMp)+1j*np.sin(IMp))
    im = np.fft.irfft2(IM)
    im[im < 0] = 0
    im[im > 1] = 1
    return im

# gerchbergSaxton() Function:
# accepts three arguments: im (array), maxIters (int), Dphi()
# applies the Gerchberg-Saxton Algorithm to the inputted image
# returns a list of images (array) which serves as the "frames" of a video
def gerchbergSaxton(im,maxIters,Dphi,mask):
    (IMa,IMp) = dft2(im)
    images = []
    errors = []
    for k in range(maxIters+1):
        print("Iteration %d of %d" % (k,maxIters))

        lt = k/(maxIters) # linear transformation factor
        phase = IMp + lt*Dphi
        
        im = idft2(IMa,phase)
        images.append(im)
        error = occultError(im,mask)
        errors.append(error)
    
    print(errors)
    return images, errors

def occultError(im,mask):
   
    error = []
    for y in range(0, im.shape[0]):
        for x in range(0, im.shape[1]):
            if mask[y,x] == True:
                error.append(im[y,x]**2)

    return sum(error)




# saveFrames() Function:
# accepts images (list of arrays containing info about the images transformed in the GS algorithm) as an argument
# exports each iteration of the images in the list as an grayscaled .png file
# the resulting .png images are used to create an .avi file in the coronaAnimate.py
def saveFrames(images, errors):
    shape = (images[0].shape[0],images[0].shape[1],3)
    image = np.zeros(shape,images[0].dtype)
    maxIters = len(images)-1

    for k in range(maxIters+1):
        image[:,:,0] = images[k]
        image[:,:,1] = images[k]
        image[:,:,2] = images[k]

        plt.subplot(1,2,1)
        y = errors
        x = np.arange(0,11)
        plt.plot(x,y,'r')


        plt.subplot(1,2,2)
        plt.imshow(image)
        plt.title(f'Iteration {k:d} of {maxIters:d}')
        plt.axis('off') #hides the labels and axes
        
        plt.savefig('coronagraph'+str(k)+'.png')
        plt.show()

main()
