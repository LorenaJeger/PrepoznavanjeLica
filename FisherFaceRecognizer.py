#Izvor: https://docs.opencv.org/3.4/da/d60/tutorial_face_main.html

import cv2
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


subjects=[]
width_d, height_d = 400, 500
detektirao_lice=0
ne_detektirano_lice=0
#funkcija za detekciju lica pomocu OpenCV
def detect_face(img):
    #Pretvoriti testne slike u sive slike jer open cv recognition tako očekuje
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
    #učitavanje OpenCV face detector cv2.CascadeClassifier, korišten je  Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')
   

    # koristimo metodu `detectMultiScale` klase `cv2.CascadeClassifier` za otkrivanje svih lica na slici
    # faces sadržava popis svih lica koja su detektirana
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
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
    i=0
    dirs = os.listdir(data_folder_path)   #Dobija putanju za izlistanje mapa u gore prosljedenom paramtru koje predstavlja svaka svoju osobu

    faces = []  #lista u kojoj se spremaju sva lica
    labels = []   #lista u kojoj se spremaju sve labele
  
    for dir_name in dirs:    #idi kroz svaki direktorij to jest svaku mapu i procitaj slike u njemu, za svaki dir_name(naziv mape u treningu) u dirs(trening mapa)
        
        # Probaj dobiti nazive imena foldera 
        if dir_name.startswith("."): continue
        if(i == 0): print("null indeks")
        print("i:", i)
        name = dir_name
        subjects.append(name)
        i_string= str(i)
        label = int(dir_name.replace(name, i_string))
        subject_dir_path = data_folder_path + "/" + dir_name    #definiranje trenutnog direktorija za trenutni subjekt subject_dir_path    #primjer sample subject_dir_path = "training-data/s1"
        subject_images_names = os.listdir(subject_dir_path)   #sprema naziv trenutne slike subjekta unutar subject direktorija

        for image_name in subject_images_names:    #za svaki naziv slike procitaj sliku u  subject_images_names
                
                if image_name.startswith("."):   #ignoriraj system files like .DS_Store
                    continue
                
                image_path = subject_dir_path + "/" + image_name   #stvori putanju slike path  #primjer image path = training-data/s1/1.pgm
                image = cv2.imread(image_path) #čitaj sliku na tom prosljedenom image_pathu
                cv2.imshow("Training on image...", cv2.resize(image, (width_d, height_d)))  #prikazati prozor slike za prikaz slike
                cv2.waitKey(100)
                face, rect = detect_face(image)   #pozivamo funkciju za detekciju lica gore definiranu i prosljedujemo joj trenutnu sliku
                if face is not None: #ako lice nije none tj, ako je pronađeno lice
                    faces.append(cv2.resize(face, (width_d, height_d)))
                    labels.append(label)  #dodaj label za to lice
        i += 1    
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
   
    
    return faces, labels, subjects
 
print("Pripremam ppodatke za treniranje...")

from datetime import datetime
start_time = datetime.now()
faces, labels, subjects = prepare_training_data("training-data")  #Pozivanjem funkcije prepare_trening_data s parametrom trening mape dobivamo 2 liste jedna sadrži sva lica druga sve labele za sva lica


print("subjects: ", subjects)
print("Podaci pripremljeni")
end_time = datetime.now()
print('Vrijeme pripreme_ podatak tj detekcija i spremanje: {}'.format(end_time - start_time))


# Treniranje Face Recognizer u ovom primjeru ćemo koristiti FisherFaceRecognizer
#Kreiramo FisherFace face recognizer 
face_recognizer = cv2.face.FisherFaceRecognizer_create()

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
  

#unkcija za pisanje imena
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2,(255, 255, 255), 1)
   


#Funkcija predikcije prepoznaje danu osobu na slici kao ulazni pamatar i crta pravokutnik oko otkrivenog lica s imenom
def predict(test_img, predvideno_ime):

    
    img = test_img.copy()  #Kopiramo sliku da sačuvamo original
    face, rect = detect_face(img)     #Detektiramo lice na slici
    
    if face is None or rect is None : 
        return None, None
     
    else:
        face = cv2.resize(face, (width_d, height_d)) # FisherFace traži da su sve slike iste velicina pa bih to navela kao nedostatak
        label, confidence = face_recognizer.predict(face)  #Predviđamo sliku pomoću face_recognizer kojeg smo trenirali prosljedujemo mu sliku 
        label_text = subjects[label]  #dobivamo naziv odgovarajuće oznake koju vraća face recognizer
        predvideno_ime=label_text
        draw_rectangle(img, rect) #crtamo pravokutnik oko detektirane slike 
        draw_text(img, label_text, rect[0], rect[1]-5)    #ispisujemo ime od predicted osobe
        return img, predvideno_ime


#Pozivanje funkcije predvidanja na testnom skupu

print("Predikcija slika u tijeku...")
start_time = datetime.now()  #sluzi da prikaz trajanja vremena

data_folder_path='test-data'
dirs = os.listdir(data_folder_path) 
nazivi=[]
y_test= []
y_pred= []
for slika in dirs:
    if slika.startswith("."):   #ignoriraj system files like .DS_Store
                    continue 
    name= slika
    subjects.append(nazivi)

    img_path= data_folder_path + "/" + name
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    predvideno_ime=""
    predict_img, predvideno_ime=predict(img, predvideno_ime)

    if(predict_img is None):
        ne_detektirano_lice= ne_detektirano_lice +1
        
    else: 
        detektirao_lice=detektirao_lice+1
        cv2.imshow("Predvidam na testu", cv2.resize(predict_img, (400, 500)))
        y_test.append(''.join([i for i in name.replace(" ", "").split(".")[0].casefold().replace("_", "").replace("test", "") if not i.isdigit()]))
        y_pred.append(predvideno_ime.replace(" ", "").replace("_", "").casefold())

    cv2.waitKey(100) 

end_time = datetime.now()
print('Vrijeme predikcije: {}'.format(end_time - start_time))

print("Classification_report")
print(classification_report(y_test, y_pred)) 

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()