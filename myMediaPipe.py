''' collection of functions useful to run mediapipe and to save videos '''
import mediapipe as mp
import cv2
import numpy as np

def findLandMarks (image):
    """
    From the image, returns the result in the typical format of cv2
    :param image:
    :return:
    """

    '''renaming variables for a better handling: ALIASING'''
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, \
                        min_tracking_confidence=0.5) as hands:
        '''Process the converted RGB image from a BGR one'''
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results

def resultsToArray(results, frameCounter):
    """
    :param results: as expressed by cv2: hands.process(...
    :param frameCounter: number of the frame
    :return: an array containing the frame number and the position of hands landmarks
    [frameNumber, x0, y0, z0, x1, y1, z1... x20, y20, z20]
    """
    '''initialize the array that will contain the values'''
    landMarkArray = [frameCounter]

    if not results.multi_hand_landmarks:
        for number in range(21 * 3):
            landMarkArray.append(np.nan)
    else:
        # todo: better solution than this HARDCODED one
        for number in range(21):
            landMarkArray.append(results.multi_hand_landmarks[0].landmark[number].x)
            landMarkArray.append(results.multi_hand_landmarks[0].landmark[number].y)
            landMarkArray.append(results.multi_hand_landmarks[0].landmark[number].z)
            # number [0] because interested in only one hand -> put max_num_hands = 1
    return landMarkArray

def defineHeader(firstColumnTitle, letters, maxNumber):
    '''Creates the header of the csv arrayOfValues: frameNumber, x0, y0, z0, x1, y1, z1 ... x20, y20, z20'''
    array = [firstColumnTitle]
    for number in range(maxNumber):
        for letter in letters:
            name = letter + "_" + str(number)
            array.append(name)
    return array

def saveData(header, arrayOfValues, CSVfileName):
    '''Opens the csv arrayOfValues specified and writes on it
    the content of ArrayOfValues, which means:
    - the frame number
    - the XYZ position of each one of the landmark'''

    frameCounter = arrayOfValues[0]
    if frameCounter == 0:
        f = open(CSVfileName, 'w', encoding='UTF8', newline='')
        writer = csv.writer(f)
        writer.writerow(header) # write the header
    else:
        f = open(CSVfileName, 'a', encoding='UTF8', newline='')
        writer = csv.writer(f)
    writer.writerow(arrayOfValues) # write in both cases the array of values
    f.close()

def drawOnImage(results, image):
    '''Inputs:
    - results as expressed by cv2: hands.process...
    - image
    Output:
    the image, (without landmarks if results are not found)
    the image with the landmarks (otherwise)'''

    '''renaming variables for a better handling: ALIASING'''
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    if not results.multi_hand_landmarks:
        return image
    else:
        image_height, image_width, _ = image.shape
        '''Copy the grabbed image'''
        annotated_image = image.copy()
        ''' A for loop for every found hand (https://stackoverflow.com/questions/67141844/how-do-i-get-the-coordinates-of-face-mash-landmarks-in-mediapipe)'''
        for hand_landmarks in results.multi_hand_landmarks:
            '''to add the drawing to the image'''
            mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())
        return annotated_image

def resultsToMatrix(results):
    """
    From the output of mediapipe.process, whose output is:
    - multi_hand_landmarks
    - multi_hand_world_landmarks
    - multi_handedness
    in which the date are given in lists, extracts a table for the first two for a better handling in the successive code
    :param results:
    :return:
    """
    w, h = 3, 21
    Matrix = np.zeros((21, 3))
    for number in range(21):
        Matrix[number][0] = results.multi_hand_landmarks[0].landmark[number].x
        Matrix[number][1] = results.multi_hand_landmarks[0].landmark[number].y
        Matrix[number][2] = results.multi_hand_landmarks[0].landmark[number].z
        # number [0] because interested in only one hand -> put max_num_hands = 1
    return Matrix