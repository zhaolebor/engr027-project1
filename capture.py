########################################################################
#
# File:   capture.py
# Author: Matt Zucker
# Date:   January, 2012 (Updated January, 2017)
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

# Figure out what input we should load:
input_device = None

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
    input_filename = 'bunny.mp4'

# Choose camera or file, depending upon whether device was set:
if input_device is not None:
    capture = cv2.VideoCapture(input_device)
    if capture:
        print('Opened camera device number', input_device, '- press Esc to stop capturing.')
else:
    capture = cv2.VideoCapture(input_filename)
    if capture:
        print('Opened file', input_filename)

# Bail if error.
if not capture or not capture.isOpened():
    print('Error opening video capture!')
    sys.exit(1)

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

filename = 'captured.'+ext

writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
if not writer:
    print('Error opening writer')
else:
    print('Opened', filename, 'for output.')
    writer.write(frame)

# Loop until movie is ended or user hits ESC:
while 1:

    # Get the frame.
    ok, frame = capture.read(frame)

    # Bail if none.
    if not ok or frame is None:
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Write if we have a writer.
    if writer:
        writer.write(grayscale)

    # Throw it up on the screen.
    cv2.imshow('Video', grayscale)

    # Delay for 5ms and get a key
    k = cv2.waitKey(5)

    # Check for ESC hit:
    if k % 0x100 == 27:
        break
