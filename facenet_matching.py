from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from mtcnn.Detection.load_mtcnn import  load_mtcnn
## facenet
import cv2
from face_encoder.facenet import Facenet
from  face_encoder.support import load_faces,predict, matching
import configparser
import argparse
import glob

def test_with_cam(facenet,facedb):
    cam = cv2.VideoCapture(0)
    mtcnn_detector = load_mtcnn(scale_factor=0.709)
    while True:
        ret, frame = cam.read()
        if ret:
            faces, _ = mtcnn_detector.detect(frame)
            for face in faces:
                x1, y1, x2, y2 = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                face_image = frame[y1:y2,x1:x2]
                names, sims = predict(face_image, facenet, facedb, VERIFICATION_THRESHOLD=0.5)
                print(names)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == 32:
                cv2.waitKey(0)
        else:
            break

def predict_image(image):
    names, sims = predict(image, facenet, face_db, VERIFICATION_THRESHOLD=0.5)
    print(names)

def get_input(frame, mtcnn):
    faces, _ = mtcnn.detect(frame)
    for face in faces:
        x1, y1, x2, y2 = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        face_image = frame[y1:y2,x1:x2]
        return face_image

def read_image_from_folder(folder_name):
    files = glob.glob(folder_name + "/*")
    img1 = cv2.imread(files[0])
    img2 = cv2.imread(files[1])
    return img1,img2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features with CNN')
    parser.add_argument('--folder', '-f', help='image folder')
    parser.add_argument('--thresh', '-th', default=0.5 , help='threshold')
    args = parser.parse_args()
    # load model
    facenet = Facenet("face_encoder/models/20180402-114759.pb")
    # conf = configparser.ConfigParser()
    # conf.read("config/global.cfg")
    # FACE_DB_PATH = conf.get("PATH", "FACE_DB_PATH")
    # face_db = load_faces(FACE_DB_PATH,facenet)

    # print(len(face_db))
    # test_with_cam(facenet,face_db)
    # image = cv2.imread("face_db/dan/0.jpg")
    # predict_image(image)
    mtcnn_detector = load_mtcnn(scale_factor=0.709)
    while True:
        img1, img2 = read_image_from_folder(args.folder)
        face_image1 = get_input(img1, mtcnn_detector)
        face_image2 = get_input(img2, mtcnn_detector)
        dist, sim = matching(face_image1, face_image2, facenet)
        print("Thresh:", args.thresh)
        print("Similarity:", sim)
        if float(args.thresh) > sim:
            print("Different person")
        else:
            print("The same person")
        print("----------------")
        cv2.imshow("img1", face_image1)
        cv2.imshow("img2", face_image2)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
#   
