import numpy
from pygame import mixer
import time
import cv2
from tkinter import *
import tkinter.messagebox
import cv2
import dlib
from scipy.spatial import distance
from scipy.spatial import distance
from PIL import Image, ImageTk
from imutils import face_utils
import tkinter as tk
import numpy as np
import pygame
import dlib
from scipy.spatial import distance as dist
from tkinter import *
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2


def exitt():
    exit()


def web():
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


def webrec():
    capture = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    op = cv2.VideoWriter('Sample1.avi', fourcc, 11.0, (640, 480))
    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        op.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    op.release()
    capture.release()
    cv2.destroyAllWindows()


def webdet():
    capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    eye_glass = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, 'Face', (x+w, y+h), font,
                        1, (250, 250, 250), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eye_g = eye_glass.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eye_g:
                cv2.rectangle(roi_color, (ex, ey),
                              (ex+ew, ey+eh), (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


def webdetRec():
    capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    eye_glass = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    op = cv2.VideoWriter('Sample2.avi', fourcc, 9.0, (640, 480))

    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, 'Face', (x+w, y+h), font,
                        1, (250, 250, 250), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eye_g = eye_glass.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eye_g:
                cv2.rectangle(roi_color, (ex, ey),
                              (ex+ew, ey+eh), (0, 255, 0), 2)
        op.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    op.release()
    capture.release()
    cv2.destroyAllWindows()


def alert():
    mixer.init()
    alert = mixer.Sound('beep-07.wav')
    alert.play()
    time.sleep(0.1)
    alert.play()


def sound_alarm():
    # play an alarm sound
    playsound.playsound("alert  sound.mp3")


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


def blink(EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES):

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--webcam", type=int, default=0,
                    help="index of webcam on system")
    args = vars(ap.parse_args())

    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold for to set off the
    # alarm

    # initialize the frame counter as well as a boolean used to
    # indicate if the alarm is going off
    COUNTER = 0
    ALARM_ON = False

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # start the video stream thread
    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=args["webcam"]).start()
    time.sleep(1.0)

    i = 0
    min_ear = 100
    max_ear = 0
    ear = 0
    # loop over frames from the video stream
    global text
    while True:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        frame = vs.read()
        frame = imutils.resize(frame, width=650)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.putText(frame, f"EYE_AR_THRESH = {EYE_AR_THRESH}", (10, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"EYE_AR_CONSEC_FRAMES = {EYE_AR_CONSEC_FRAMES}", (300, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1

                # if the eyes were closed for a sufficient number of
                # then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    # if the alarm is not on, turn it on
                    if not ALARM_ON:
                        ALARM_ON = True

                        # check to see if an alarm file was supplied,
                        # and if so, start a thread to have the alarm
                        # sound played in the background
                        t = Thread(target=sound_alarm)
                        t.deamon = True
                        t.start()

                    # draw an alarm on the frame
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
            else:
                COUNTER = 0
                ALARM_ON = False

            # draw the computed eye aspect ratio on the frame to help
            # with debugging and setting the correct eye aspect ratio
            # thresholds and frame counters
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (500, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        if i < 50:
            if ear < min_ear:
                min_ear = ear
            elif ear > max_ear:
                max_ear = ear
        elif i == 50:
            EYE_AR_THRESH = (min_ear + max_ear)/2
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


window = Tk()

window.geometry("1355x768+0+0")
window.configure(bg="#ffffff")
canvas = Canvas(
    window,
    bg="#ffffff",
    height=800,
    width=1209,
    bd=0,
    highlightthickness=0,
    relief="ridge")
canvas.place(x=0, y=0)

background_img = PhotoImage(
    file=f"C:\\Users\\krunal\\Desktop\\project related stuffs\\New folder\\background.png")
background = canvas.create_image(
    330, 350,
    image=background_img)

img0 = PhotoImage(
    file=f"C:\\Users\\krunal\\Desktop\\project related stuffs\\New folder\\img0.png")
b0 = Button(
    image=img0,
    borderwidth=0,
    highlightthickness=0,
    command=exitt,
    relief="flat")

b0.place(
    x=830, y=600,
    width=259,
    height=83)

img1 = PhotoImage(
    file=f"C:\\Users\\krunal\\Desktop\\project related stuffs\\New folder\\img1.png")
b1 = Button(
    image=img1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: blink(0.31, 48),
    relief="flat")

b1.place(
    x=700, y=500,
    width=511,
    height=87)

img2 = PhotoImage(
    file=f"C:\\Users\\krunal\\Desktop\\project related stuffs\\New folder\\img2.png")
b2 = Button(
    image=img2,
    borderwidth=0,
    highlightthickness=0,
    command=webdetRec,
    relief="flat")

b2.place(
    x=700, y=400,
    width=515,
    height=88)

img3 = PhotoImage(
    file=f"C:\\Users\\krunal\\Desktop\\project related stuffs\\New folder\\img3.png")
b3 = Button(
    image=img3,
    borderwidth=0,
    highlightthickness=0,
    command=webdet,
    relief="flat")

b3.place(
    x=700, y=300,
    width=516,
    height=77)

img4 = PhotoImage(
    file=f"C:\\Users\\krunal\\Desktop\\project related stuffs\\New folder\\img4.png")
b4 = Button(
    image=img4,
    borderwidth=0,
    highlightthickness=0,
    command=webrec,
    relief="flat")

b4.place(
    x=700, y=200,
    width=523,
    height=83)

img5 = PhotoImage(
    file=f"C:\\Users\\krunal\\Desktop\\project related stuffs\\New folder\\img5.png")
b5 = Button(
    image=img5,
    borderwidth=0,
    highlightthickness=0,
    command=web,
    relief="flat")

b5.place(
    x=700, y=100,
    width=522,
    height=82)

window.resizable(False, False)
window.mainloop()
