# Izvor: https://github.com/ageitgey/face_recognition

import face_recognition
import sklearn 
from sklearn import svm
import os
from sklearn.metrics import classification_report


# Training the SVC classifier

# Trening podaci -> faces polje slika sa prepoznatim licima enkodiranima, names polje sadrzava imena tih lica
faces = []
names = []


# Training direktorij
dirs = os.listdir("/Users/lorena/Documents/PrepoznavanjeLicaKod/training-data")

data_folder_path = ("training-data")


from datetime import datetime
start_time = datetime.now()  #sluzi da prikaz trajanja vremena

for dir_name in dirs:  
    if dir_name.startswith("."): continue
    subject_dir_path = data_folder_path + "/" + dir_name    #definiranje trenutnog direktorija za trenutni subjekt subject_dir_path    
    subject_images_names = os.listdir(subject_dir_path) 
    for image_name in subject_images_names:    #za svaki naziv slike procitaj sliku u  subject_images_names
                
                if image_name.startswith("."):   #ignoriraj system files like .DS_Store
                    continue
                
                image_path = subject_dir_path + "/" + image_name   #stvori putanju slike path  
               
       
                face = face_recognition.load_image_file(image_path)
                face_bounding_boxes = face_recognition.face_locations(face)

                 #Ako trening slika sadrzi jedno lice
                if len(face_bounding_boxes) == 1:
                    face_enc = face_recognition.face_encodings(face)[0]
                    
                    #Dodaj kodiranje lica za trenutnu sliku s odgovarajućom oznakom (imenom) u podatke o treningu
                    #dodaj lice (face) u listu lica(faces)
                    #dodaj ime za to lice
                    faces.append(face_enc)
                    names.append(subject_dir_path)
                else:
                    print("Br lica je razlicit od 1")

end_time = datetime.now()
print('Vrijeme pripreme podataka: {}'.format(end_time - start_time))

# Create and train the SVC classifier
from datetime import datetime
start_time = datetime.now()  #sluzi da prikaz trajanja vremena

clf = svm.SVC(gamma='scale')
clf.fit(faces,names)

end_time = datetime.now()
print('Vrijeme kreiranja i treniranje SCV klasifikatora: {}'.format(end_time - start_time))


data_folder_path='test-data'
dirs = os.listdir(data_folder_path) 

from datetime import datetime
start_time = datetime.now()  #sluzi da prikaz trajanja vremena
y_test= []
y_pred= []

for slika in dirs:
    if slika.startswith("."):   
                    continue 
    naziv= slika
    img_path= data_folder_path + "/" + naziv

    # Učitajte testnu sliku s nepoznatim licima u numpy niz
    test_image = face_recognition.load_image_file(img_path)

    # Nađi sva lica na testnoj slici koristeći  HOG-based model
    face_locations = face_recognition.face_locations(test_image)
    no = len(face_locations)
    print("Number of faces detected: ", no)


# Predikcija svih lica u trenutnoj slici testa koristeći trenirani klasifikator
    print("Found:")
    for i in range(no):
          test_image_enc = face_recognition.face_encodings(test_image)[i]
          predvideno_ime = clf.predict([test_image_enc])
          print(*predvideno_ime)
          print(predvideno_ime)
          y_test.append(''.join([i for i in naziv.replace(" ", "").split(".")[0].casefold().replace("_", "").replace("test", "") if not i.isdigit()]))
          y_pred.append(predvideno_ime[0].lstrip("training-data/").replace(" ", "").replace("_", "").casefold())
         

print(classification_report(y_test, y_pred)) 

end_time = datetime.now()
print('Vrijeme predvidanja: {}'.format(end_time - start_time))




