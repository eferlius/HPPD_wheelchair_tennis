'''To display in real time the application of MediaPipe engine for the hand recognition'''

# ----------------------------- IMPORT
import cv2
import keyboard
import myMediaPipe as mpp
import time
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------- PARAMETERS

# ----------------------------- FLAGS

# ----------------------------- FUNCTIONS

# ----------------------------- MAIN
if __name__ == '__main__':
    # initialization of the camera
    cap = cv2.VideoCapture(0)
    print('\nStarting the camera...')

    # reading the first image
    success, image = cap.read()
    image = cv2.flip(image, 1)

    #setting the frame counter
    frameCounter = 0
    fig = plt.figure()
    plt.ion()
    while success: #the camera is still sending images
        tic = time.time() # to log information about execution

        # finding the landmarks on the acquired image
        results = mpp.findLandMarks(image)

        # creating a new image with the drawing of the landmarks on it
        annotated_image = mpp.drawOnImage(results, image)

        # showing the image
        cv2.imshow("tmp", annotated_image)
        cv2.waitKey(1) #add waitKey when displaying an image

        # logging information about this execution
        toc = time.time()
        elapsed = toc - tic
        print('frame:  ' + str(frameCounter) + ' // elapsed: ' + "%.2f" % elapsed)

        # preparing for the next execution
        frameCounter = frameCounter + 1

        # reading the next image
        success, image = cap.read()
        image = cv2.flip(image, 1)

        if results.multi_hand_landmarks:
            # 3d rendering
            w, h = 3, 21
            Matrix = np.zeros((21, 3))
            for number in range(21):
                Matrix[number][0] = results.multi_hand_landmarks[0].landmark[number].x
                Matrix[number][1] = results.multi_hand_landmarks[0].landmark[number].y
                Matrix[number][2] = results.multi_hand_landmarks[0].landmark[number].z

            plt.pause(0.05)
            ax = plt.axes(projection='3d')
            ax.stem(Matrix[:, 0], Matrix[:, 1], Matrix[:, 2])

        if keyboard.is_pressed("q"):
            print("q pressed, ending loop")
            break

    cap.release()

    print(results.multi_hand_landmarks)