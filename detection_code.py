import cv2
import numpy as np
import pywt
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib as mpl

def readImage(imagePath):
    # Read the input image, resize it and convert it into grayscale
    inputImage = cv2.imread(imagePath)
    inputImage = cv2.resize(inputImage, (512, 512))
    grayImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    return inputImage, grayImage

def SWT(grayImage,wavelet):
    # Considering only the approximation coefficients that contain most of the information
    reducedImage=pywt.swt2(grayImage,wavelet,level=2,trim_approx=True)
    return reducedImage

def keypoints(reducedImage):
    # Extract keypoints obtained by the SIFT method
    sift=cv2.SIFT.create()
    keypts,descp = sift.detectAndCompute(reducedImage,None)
    return keypts,descp

def showImage(title, image):
    # Display the image in a separate dialog box
    cv2.imshow(title, image)

def showKeypoints(inputImage, features, colour):
    # Display an image with the extracted keypoints highlighted
    img1=inputImage.copy()
    img = cv2.drawKeypoints(img1, features, None, color=colour)
    return img

def dbscan(kp,dp,img):
    # First level DBSCAN for getting all the keypoint pairs that are similar
    db=DBSCAN(eps=100,min_samples=2,metric='euclidean')
    db.fit(dp)
    labels = db.labels_
    kp = kp[labels != -1]
    labels = labels[labels != -1]

    if len(kp) < 5:
        # Number of level 1 samples less than 10, hence the forged regions are not detected by this algo
        return ([],[])

    # Second-level DBSCAN for grouping the closely located keypoints, highlighting the original and forged region pairs
    img_cls = DBSCAN(eps=20,min_samples=8,metric='euclidean')
    img_cls.fit(kp)
    lbl = img_cls.labels_
    unique_labels = list(set(lbl))
    print(lbl)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for i in range(len(colors)):
        colors[i] = mpl.colors.rgb2hex(colors[i])
    print(colors[0], len(unique_labels), len(lbl))

    img = plt.imread(img)
    fig, ax = plt.subplots()
    ax.imshow(img,extent=[0,512,512,0])

    #Visualization of final localization results
    for i in range(len(kp)):
        if lbl[i] == -1:
            pass
        else:
            col_idex = unique_labels.index(lbl[i])
            ax.scatter(kp[i, 0], kp[i, 1], color=colors[col_idex])
    plt.show()

def drivercode(path):
    inputImage, grayImage = readImage(path)
    reducedImage = SWT(grayImage, 'haar')
    ll = reducedImage[0].astype(np.uint8)
    kp, dp = keypoints(ll)
    pts = cv2.KeyPoint_convert(kp)
    dbscan(pts, dp, path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    folder_path = "D:/Soham Maji Project/New folder/Input images/CoMoFoD_50/0"
    for i in range(1,26):
        print("_______________________________\n\nImage", i)

        img_path = folder_path + str(i // 10) + str(i % 10) + "_F.png"      #+a
        drivercode(img_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
