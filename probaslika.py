
import cv2
import glob

imdir = 'test-data'
ext = ['png', 'jpg', 'jpeg']    # Add image formats here

files = []
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

images = [cv2.imread(file, cv2.IMREAD_COLOR) for file in files]




cv2.imshow("Proba", images)
 
cv2.waitKey(0)

cv2.destroyAllWindows()