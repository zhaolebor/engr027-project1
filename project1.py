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
import os
from matplotlib import pyplot as plt

matplotlib.interactive(True)


frames = []
new_frames = []
displays = []
all_points = []
white = (255,255,255)


# Figure out what input we should load:
input_device = None

######################################################################
# Helper function to pick points
def pickPoints(window, image, filename, xcoord=0):

    cv2.namedWindow(window)
    cv2.imshow(window, image)
    cv2.moveWindow(window, xcoord, 0)

    w = cvk2.MultiPointWidget()

    if w.load(filename):
        print('loaded points from {}'.format(filename))
    else:
        print('could not load points from {}'.format(filename))

    ok = w.start(window, image)

    if not ok:
        print('user canceled instead of picking points')
        sys.exit(1)

    w.save(filename)

    return w.points

# Helper function to calculate distance between two points
def distance(p0, p1):
    return numpy.sqrt(numpy.square(p0[0] - p1[0]) + numpy.square(p0[1] - p1[1]))

# Helper function to display the image until a key event
def fixKeyCode(code):
    return numpy.uint8(code).view(numpy.int8)

######################################################################
# Main program starts here
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

basename = os.path.basename(input_filename)
prefix, _ = os.path.splitext(basename)
datafile = prefix + '.txt'

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

# Prompt user input for parameters
print('finished reading')
print()
threshold_value = int(raw_input('Enter an int threshold value: '))
erosion_kernel_size = int(raw_input('Enter an int erosion kernel size: '))
dilation_kernel_size = int(raw_input('Enter an int dilation kernel size: '))
temp_width = int(raw_input('Enter an int temporal average width: '))


# Begin processing
for i in range(len(frames)-1):
    sum = numpy.zeros_like(frames[i], dtype='uint32')
    for j in range(temp_width):
        if i+j >= len(frames):
            sum += frames[len(frames)-1]
        else:
            sum += frames[i+j]
    old_average = sum/temp_width
    average = numpy.array(old_average, dtype='uint8')

    new_frame = cv2.absdiff(average, frames[i])
    ret,thresh1 = cv2.threshold(new_frame, threshold_value, 255, cv2.THRESH_BINARY)

    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple((erosion_kernel_size,erosion_kernel_size)))
    erosion = cv2.erode(thresh1, kernel_erosion)

    kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple((dilation_kernel_size,dilation_kernel_size)))
    dilation = cv2.dilate(erosion, kernel_dilation)

    # Save thresholded images before and after morph ops, also save original frame for reference
    if i == 0 or i == len(frames)-2 or i == len(frames)/2:
        cv2.imwrite('pre-morph-op_at_frame_'+ str(i) +'.jpg', thresh1)
        cv2.imwrite('post_morph_op_at_frame_'+ str(i) +'.jpg', dilation)
        cv2.imwrite('original_at_frame_'+ str(i) +'.jpg', frames[i])



    work = dilation.copy()
    image, contours, hierarchy = cv2.findContours(work, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    display = numpy.zeros((dilation.shape[0], dilation.shape[1], 3), dtype='uint8')
    new_points = []
    for j in range(len(contours)):
        info = cvk2.getcontourinfo(contours[j])
        mu = info['mean']

        mu_new = cvk2.array2cv_int(mu)
        new_points.append(mu_new)

        cv2.circle(display, cvk2.array2cv_int(mu), 3, white, 1, cv2.LINE_AA)

    all_points.append(new_points)

    displays.append(display)

    dilation_rgb = cv2.cvtColor(dilation, cv2.COLOR_GRAY2RGB)

    new_frames.append(dilation_rgb)

for i in range(len(all_points)):
    print(len(all_points[i]))

w = new_frames[0].shape[1]
h = new_frames[0].shape[0]

print(w,h)

print('finished processing, saved pre and post morph op images at frame 0, ' + str(len(frames)/2) + ', ' + str(len(frames)-2))
print()

# Begin tracking process
print('Please pick the object you want to track from the first frame by right clicking.')
points_to_track = pickPoints('frame 0', displays[0], datafile)
print('got points =\n', points_to_track)

best_points = []
for point1 in points_to_track:
    point1 = tuple(point1[0])
    best = None
    best_distance = float('inf')
    for point2 in all_points[0]:
        curr_distance = distance(point1, point2)
        if curr_distance < best_distance:
            best = point2
            best_distance = curr_distance
    best_points.append(best)
print('found points at frame 0', best_points)

trajectory = []
trajectory.append(best_points)
for i in range(len(frames)-1):
    best_points = []
    for old_point in trajectory[0]:
        best = None
        best_distance = float('inf')
        for new_point in all_points[i]:
            curr_distance = distance(old_point, new_point)
            if curr_distance < best_distance and new_point not in best_points:
                best = new_point
                best_distance = curr_distance
        best_points.append(best)
    trajectory.append(best_points)

for j in range(len(best_points)):
    best_x = []
    best_y = []
    for i in range(len(trajectory)):
        if i == 0:
            last_point = best_points[j]
        else:
            last_point = tuple((best_x[-1], best_y[-1]))
        curr_distance = distance(last_point, trajectory[i][j])
        if curr_distance > 50:
            increment = 1
            while 1:
                if i+increment >= len(trajectory):
                    break
                else:
                    curr_distance = distance(last_point, trajectory[i+increment][j])
                    if curr_distance > 50:
                        increment += 1
                    else:
                        good_point = trajectory[i+increment][j]
                        break
            best_x.append(good_point[0])
            best_y.append(good_point[1])
        else:
            best_x.append(trajectory[i][j][0])
            best_y.append(trajectory[i][j][1])

    plt.figure(j+1)
    plt.plot(best_x, best_y, 'r-')
    plt.title('trajectory of point ' + str(j+1))
    plt.xlabel('x-pos')
    plt.ylabel('y-pos')
    plt.axis([0, w, h, 0])
    plt.draw()
    plt.savefig('trajectory_' + str(j+1) +'.png')



######################################################################
# Now set up a VideoWriter to output video.
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
plt.show()
