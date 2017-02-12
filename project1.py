########################################################################
#
# File:   project1.py
# Author: Alan Zhuolun Zhao, Pepper Shiqin Wang
# Date:   February, 2017
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# This program demonstrates how to use the VideoCapture and
# VideoWriter objects from OpenCV.
#
# Usage: the program can be run with a filename or a single integer as
# a command line argument.  Integers are camera device ID's (usually
# starting at 0).  If no argument is given, tries to capture from
# the default input 'bunny.mp4'

# Do Python 3-style printing
from __future__ import print_function

import cv2
import numpy
import sys
import struct
import cvk2
import matplotlib
from matplotlib import pyplot as plt


frames = []
new_frames = []
displays = []
all_points = []
white = (255,255,255)


# Figure out what input we should load:
input_device = None

def fixKeyCode(code):
    return numpy.uint8(code).view(numpy.int8)

if len(sys.argv) > 1:
    input_filename = sys.argv[1]
    try:
        input_device = int(input_filename)
    except:
        pass
else:
    print('Using default input. Specify a device number to try using your camera, e.g.:')
    print()
    print('  python', sys.argv[0], '0')
    print()
    input_filename = 'traffic2.mp4'

# Choose camera or file, depending upon whether device was set:
if input_device is not None:
    capture = cv2.VideoCapture(input_device)
    if capture:
        print('Opened camera device number', input_device, '- press Esc to stop capturing.')
else:
    capture = cv2.VideoCapture(input_filename)
    if capture:
        print('Opened file for reading', input_filename)

# Bail if error.
if not capture or not capture.isOpened():
    print('Error opening video capture!')
    sys.exit(1)

while 1:
    ok, frame = capture.read()
    if not ok or frame is None:
        break
    grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    frames.append(grayscale)
    k = cv2.waitKey(5)
    if k % 0x100 == 27:
        break

print('finished reading')
print()
threshold_value = int(raw_input('Enter an int threshold value: '))
erosion_kernel_size = int(raw_input('Enter an int erosion kernel size: '))
dilation_kernel_size = int(raw_input('Enter an int dilation kernel size: '))

for i in range(len(frames)):
    #if i == 0 or i == 1:
        #average = (frames[i]+frames[i+1]+frames[i+2])/3
    #elif i == len(frames)-2 or i == len(frames)-1:
        #average = (frames[i]+frames[i-1]+frames[i-2])/3
    if i == len(frames)-1:
        average = (frames[i-1]+frames[i])/2
    else:
        #average = (frames[i-1]+frames[i]+frames[i+1])/3
        average = (frames[i]+frames[i+1])/2
    new_frame = cv2.absdiff(average, frames[i])
    ret,thresh1 = cv2.threshold(new_frame, threshold_value, 255, cv2.THRESH_BINARY)

    if i == 0 or i == len(frames)-1 or i == len(frames)/2:
        cv2.imwrite('pre-morph-op_at_frame_'+ str(i) +'.jpg', thresh1)

    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple((erosion_kernel_size,erosion_kernel_size)))
    erosion = cv2.erode(thresh1, kernel_erosion)

    kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple((dilation_kernel_size,dilation_kernel_size)))
    dilation = cv2.dilate(erosion, kernel_dilation)

    if i == 0 or i == len(frames)-1 or i == len(frames)/2:
        cv2.imwrite('post_morph_op_at_frame_'+ str(i) +'.jpg', dilation)

    work = dilation.copy()
    image, contours, hierarchy = cv2.findContours(work, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    display = numpy.zeros((dilation.shape[0], dilation.shape[1], 3), dtype='uint8')

    new_points = []
    for j in range(len(contours)):
        info = cvk2.getcontourinfo(contours[j])
        mu = info['mean']

        mu_new = cvk2.array2cv_int(mu)
        new_points.append(mu_new)

        #plt.plot(mu)
        #plt.xlabel('x-pos')
        #plt.ylabel('y-pos')
        #plt.show

        cv2.circle(display, cvk2.array2cv_int(mu), 3, white, 1, cv2.LINE_AA)

    all_points.append(new_points)
    for i in range(len(all_points)):
        print(len(all_points[i]))
    #old_points = new_points
    #w = cvk2.MultiPointWidget()
    #w.points = points
    #print(w.points)

    displays.append(display)

    new_frames.append(dilation)

print('finished processing, saved pre and post morph op images at frame 0, ' + str(len(frames)/2) + ', ' + str(len(frames)-1))
print()

capture = cv2.VideoCapture(input_filename)
if capture:
    print('Opened file again', input_filename)

# Fetch the first frame and bail if none.
ok, frame = capture.read()

if not ok or frame is None:
    print('No frames in video')
    sys.exit(1)

# Now set up a VideoWriter to output video.
w = frame.shape[1]
h = frame.shape[0]

fps = 30

# One of these combinations should hopefully work on your platform:
fourcc, ext = (cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 'avi')
#fourcc, ext = (cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 'mov')

filename = 'project1.'+ext

writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
if not writer:
    print('Error opening writer')
else:
    print('Opened', filename, 'for output.')
    #writer.write(thresh1)

# Loop until movie is ended or user hits ESC:
for i in range(len(new_frames)):

    # Write if we have a writer.
    if writer:
        writer.write(new_frames[i])

    # Throw it up on the screen.
    #cv2.imshow('Video', new_frames[i])
    cv2.imshow('Means', displays[i])

    # Delay for 5ms and get a key
    k = cv2.waitKey(5)

    # Check for ESC hit:
    if k % 0x100 == 27:
        break
