# haar cascade face detection

# imports
import cv2
import customClassifier # my custom classifier
import json

from flask import Flask, Response

vidstream = Flask(__name__)

camInt = 0 # 0 is internal, 1 is external

# function to get username
def get_username(nonEmptyFile, unames):
    print("NEW ID!")
    while True:
        try:
            username = input("Enter a new username... (make it unique) ").lower()
            if username == '':
                raise Exception
            if nonEmptyFile and username in unames:
                raise Exception
        except:
            print("MUST BE NON-EMPTY and NEW!")
        else:
            print("Username: "+username)
            return username

# ask if user wants to train the model
while True:
    try:
        user_trainCase_text = input("Would you like to train the model? (yes/no)")
        user_trainCase_text = user_trainCase_text.lower()
        if user_trainCase_text != 'yes' and user_trainCase_text != 'no':
            raise Exception
    except:
        print("INVALID CHOICE!")
    else:
        if user_trainCase_text == 'yes':
            user_trainCase = True
        else:
            user_trainCase = False
        break

# only ask for id if user wants to train the model
if user_trainCase:      
    while True:
        try:
            user_id = int(input("Enter a User ID Before we Begin... (old IDs will not accept new names) "))
            if user_id < 0:
                raise Exception
        except:
            print("INVALID TYPE")
        else:
            print("user_id: "+str(user_id)+" chosen.")
            break

# json file write and reading, setting new lists
unames = []
uids = []
file = {}

# Try to open the file and collect usernames in lists
try:
    with open('participant_data.json', 'r') as data:
        file = json.load(data)
        for person in file["users"]:
            unames.append(person['name'])
            uids.append(person['id'])
        
        if user_trainCase:
            # Only add a new entry if the id isn't already used, otherwise update name
            if user_id in uids:
                print("EXISTING ID CHOSEN! NAME: "+unames[uids.index(user_id)])
                # unames[uids.index(user_id)] = username
                # file['users'][uids.index(user_id)]['name'] = username
            
            # get username and check that a name isn't used twice
            else:
                username = get_username(True, unames)

                # json dictionary for current user again with new name
                currentUser = {"name":username, "id":user_id}
                file['users'].append(currentUser)

# If file doesn't exist, create a new one and add the only entry to the list
except (FileNotFoundError, json.JSONDecodeError):
    username = get_username(False, [])
    file = {'users': [{"name":username, "id":user_id}]} # may cause issues as list
    unames.append(currentUser['name'])
    uids.append(user_id)

# Write the updated data back to the file
with open('participant_data.json', 'w') as output:
    json.dump(file, output, indent=4)

print("\nNOTE: PRESS \'B\' TO END APPLICATION")

# creating dataset by writing ROI images (region of interest)
def gen_dataset(img, id, img_id):
    cv2.imwrite("dataset/user."+str(id)+"."+str(img_id)+".jpg", img) # making path to put cropped faces, subfolder in dataset with IDs and JPG filetype



# boundary function taking frame and classifier (eyes, nose, mouth) + custom classifier
def boundary(img, classifier, scaleFac, minNeighbours, color, text, clf):

    # make grayscale
    graySample = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # will take all faces in image, scaling to face size, with not allowing a face with insufficient regions
    features = classifier.detectMultiScale(graySample, scaleFac, minNeighbours) # will return list of features

    # stores face coords
    coords = []

    for (x, y, w, h) in features:

        # init rectangle
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)

        if not user_trainCase:
            # ignore config value, just take id, indexes crop the image to the boxes
            id, _ = clf.predict(graySample[y:y+h, x:x+w])

            # image, text, coords, font, size, color, thickness, line type, taking the json's id's corresponding name
            cv2.putText(img, unames[uids.index(id)], (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA) # trained name
        else:
            # image, text, coords, font, size, color, thickness, line type
            cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

        coords = [x, y, w, h]
    
    return coords # , img



# detect function for trained model
def recognize(img, clf, faceCasc):
    # BGR
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "magenta":(255,0,255)}
    coords = boundary(img, faceCasc, 1.1, 10, color["green"], "Face", clf)
    return img



# detection function with frame, classifier
def detect(img, faceCasc, eyeCasc, img_id):
    
    # BGR
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "magenta":(255,0,255)}

    # Use other function that was made
    coords = boundary(img, faceCasc, 1.1, 10, color["red"], "NEW FACE", clf) # coords, img =

    # if enough coordinates are there for the existence of a face cut image to crop to the face
    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]] # region of interest for eyes n glasses
        
        gen_dataset(roi_img, user_id, img_id)

        # coords = boundary(roi_img, eyeCasc, 1.1, 14, color["blue"], "EYES") # TUNE FOR EYE DETECT

    return img



# set cascades to xml files
faceCasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCasc = cv2.CascadeClassifier("haarcascade_eye.xml")
glassesCasc = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")



# import custom model
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("customClassifier.yml")



# -1 or 1 for external cam, default is 0 for webcam
stream = cv2.VideoCapture(camInt)

img_id = 0 # incremented for every image

# if no stream possible, say so an end
if not stream.isOpened():
    print("No stream found :(")
    exit()


def framify():

    global img_id

    # take frame and return boolean constantly
    while True:
        ret, frame = stream.read()

        # if stream ends print that it did
        if not ret:
            print("Stream has ended, False Return Val.")
            break

        # use functions before
        if user_trainCase:
            # no title since would interfere with captures
            frame = detect(frame, faceCasc, eyeCasc, img_id)
        else:
            # make title
            cv2.rectangle(frame, (0, 10), (260, 45), (255, 255, 255), -1)
            cv2.putText(frame, "Haar Cascade Detection Feed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 175, 0), 1)
            
            # get frame
            frame = recognize(frame, clf, faceCasc)

        # show feed
        # cv2.imshow("Facial Recognition (HAAR)", frame)

        # code to make flask-readable
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yielding the frame in the correct format for Flask streaming
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


        img_id += 1

        # stop if 'b' pressed
        if cv2.waitKey(1) == ord("b"):
            break



    # end
    stream.release()
    cv2.destroyAllWindows()
    print("Face Detection app closed successfully.")

@vidstream.route('/feed')
def feed():
    return Response(framify(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@vidstream.route('/')
def index():
    return "Haar Cascade Detection Stream"

# Running the Flask app
if __name__ == '__main__':
    vidstream.run(host='0.0.0.0', port=5000)

if user_trainCase:
    customClassifier.train("dataset") # train the custom classifier on the dataset folder