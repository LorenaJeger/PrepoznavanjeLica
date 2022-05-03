
import cv2
import os
import numpy as np


#there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "Ramiz Raja", "Elvis Presley"]


#funkcija za detekciju lica pomocu OpenCV
def detect_face(img):
    #Pretvoriti testne slike u sive slike jer open cv recognition tako očekuje
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
    #učitavanje OpenCV face detector cv2.CascadeClassifier, korišten je  LBP koji je brži
    #Također ima i Haar classifier, ali je sporiji
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

   

    # koristimo metodu `detectMultiScale` klase `cv2.CascadeClassifier` za otkrivanje svih lica na slici
    # faces sadržava popis svih lica koja su detektirana
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #Ako lice nije definirano vrati originalnu sliku
    if (len(faces) == 0):
        return None, None
    
    # Kako su lica koja vraća metoda `detectMultiScale` zapravo pravokutnici (x, y, širina, visina), 
    # a ne slike stvarnih lica, tako da moramo izdvojiti područje slike lica iz glavne slike. 
    
    #izvlačimo područje lica, pod pretpostavkom da na slici imamo 1 lice
    (x, y, w, h) = faces[0]
    
    # izdvajamo područje lica iz sive slike, vraćamo  područje slike lica i pravokutnik lica.
    
    return gray[y:y+w, x:x+h], faces[0]


# Funkcija prepare_training_data čita slike u trening skupu koji prima kao parametar,  
# Za svaku sliku u treningu čita sliku za nju detektira lice sa slike te vraća dva polja  iste veličine.
# Jedan popis će biti lista lica, a drugi drugi popis oznaka za svako lice


def prepare_training_data(data_folder_path):

    dirs = os.listdir(data_folder_path)   #Dobija putanju za izlistanje mapa u gore prosljedenom paramtru koje predstavlja svaka svoju osobu
    faces = []  #lista u kojoj se spremaju sva lica
    labels = []   #lista u kojoj se spremaju sve labele
    
   
    for dir_name in dirs:    #idi kroz svaki direktorij to jest svaku mapu i procitaj slike u njemu, za svaki dir_name(naziv mape u treningu) u dirs(trening mapa)
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name    #definiranje trenutnog direktorija za trenutni subjekt subject_dir_path    #primjer sample subject_dir_path = "training-data/s1"
        subject_images_names = os.listdir(subject_dir_path)   #sprema naziv trenutne slike subjekta unutar subject direktorija

        for image_name in subject_images_names:    #za svaki naziv slike procitaj sliku u  subject_images_names
                if image_name.startswith("."):   #ignoriraj system files like .DS_Store
                    continue;
                image_path = subject_dir_path + "/" + image_name   #stvori putanju slike path  #primjer image path = training-data/s1/1.pgm
                image = cv2.imread(image_path) #čitaj sliku na tom prosljedenom image_pathu
                cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))  #prikazati prozor slike za prikaz slike
                cv2.waitKey(100)
                face, rect = detect_face(image)   #pozivamo funkciju za detekciju lica gore definiranu i prosljedujemo joj trenutnu sliku
                if face is not None: #ako lice nije none tj, ako je pronađeno lice
                    faces.append(face)  #dodaj lice (face) u listu lica(faces)
                    labels.append(label)  #dodaj label za to lice
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels
 

print("Pripremam ppodatke za treniranje...")
from datetime import datetime
start_time = datetime.now()
faces, labels = prepare_training_data("training-data")  #Pozivanjem funkcije prepare_trening_data s parametrom trening mape dobivamo 2 liste jedna sadrži sva lica druga sve labele za sva lica
print("Podaci pripremljeni")
end_time = datetime.now()
print('Vrijeme pripreme_ podatak tj detekcija i spremanje: {}'.format(end_time - start_time))

#Ispisujemo koliko je detektirano lica i koliko je detektirano labela prilikom prpiremanja podataka, funkcija pripremi podatke poziva funkciju detekcija lica 
print("Duljina faces: ", len(faces))
print("Duljina labels: ", len(labels))



# Treniranje Face Recognizer u ovom primjeru ćemo koristiti LBPH face recognizer

# 1. EigenFace Recognizer: This can be created with `cv2.face.createEigenFaceRecognizer()`
# 2. FisherFace Recognizer: This can be created with `cv2.face.createFisherFaceRecognizer()`
# 3. Local Binary Patterns Histogram (LBPH): This can be created with `cv2.face.LBPHFisherFaceRecognizer()`



#Kreiramo LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#or use EigenFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.EigenFaceRecognizer_create()

#or use FisherFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.FisherFaceRecognizer_create()

# Treniramo face recognizer na našem trening skupu lica

from datetime import datetime
start_time = datetime.now()  #sluzi da prikaz trajanja vremena

face_recognizer.train(faces, np.array(labels))  # treniramo face_recognizer kojem prosljedujemo polje lica i numpy polje labele buduci da face recognition ocekuje da vektor oznaka bude niz
end_time = datetime.now()
print('Vrijeme treniranje_ face_recognizer: {}'.format(end_time - start_time))



# Predviđanje



# Funkcija za crtanje pravokutnika 
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
  

#funkcija za pisanje imena
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 0.8,(255, 255, 255), 1)
   


#Funkcija predikcije prepoznaje danu osobu na slici kao ulazni pamatar i crta pravokutnik oko otkrivenog lica s imenom
def predict(test_img):
    
    img = test_img.copy()  #Kopiramo sliku da sačuvamo original
    face, rect = detect_face(img)     #Detektiramo lice na slici
    
    #predict the image using our face recognizer 
    
    label, confidence = face_recognizer.predict(face)  #Predviđamo sliku pomoću face_recognizer kojeg smo trenirali prosljedujemo mu sliku 
    label_text = subjects[label]  #dobivamo naziv odgovarajuće oznake koju vraća face recognizer
    
    draw_rectangle(img, rect) #crtamo pravokutnik oko detektirane slike 
    draw_text(img, label_text, rect[0], rect[1]-5)    #ispisujemo ime od predicted osobe
    return img

#Pozivanje funkcije predvidanja na testnom skupu.

print("Predikcija slika u tijeku...")

#load test images
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")

# izvrsi predvidanje
from datetime import datetime
start_time = datetime.now()  #sluzi da prikaz trajanja vremena
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
end_time = datetime.now()
print('Predvidanje zavrseno{}'.format(end_time - start_time))

#pokazi slike predikcije
cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()


# Mjerenja: 
# Recognition Rate is the total number of correctly identified probe images, divided by the total number of probe images.
# Verification Rate. It relies on a list of image pairs, where pair with the same and pairs with different identities are compared. Given the lists of similarities of both types, the Receiver Operating
# Recognition Accuracy = (Number of recognized face images/Total Number of Face Images tested)X100