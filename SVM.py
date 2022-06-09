import face_recognition
from sklearn import svm
import os


# Training the SVC classifier

# Trening podaci -> faces polje slika sa prepoznatim licima enkodiranima, names polje sadrzava imena tih lica
faces = []
names = []
true_positive=0
true_negative=0

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


def usporedbaImena(naziv, predvideno_ime):
    #naziv = naziv slike u testu, predvideno_ime= ime koje algoritam predvida naziv mape u trening data
    #Predvideno ime training-data/Ramiz Raja
    tocnost=0
    predvideno_ime_strip= predvideno_ime.lstrip("training-data/")
    predvidenoIme_spojeno= "".join(predvideno_ime_strip.split())  #Spoji ime sa prezimenom bez razmaka 
    if naziv.endswith('.jpg'):
        name_bez_nastavka = naziv.strip(".jpg")    #iz test slike makni nastavak
    if naziv.endswith('.jpeg'):
        name_bez_nastavka = naziv.strip(".jpeg")
        
    if(predvidenoIme_spojeno.casefold() in name_bez_nastavka.casefold()):   #Da li je predvideno ime cafefold(case sensitive) sadržan u name_bez_nastavka
        tocnost=1 
        print("Nasao sam")
        return tocnost
    else : print("Nisam nasao", predvidenoIme_spojeno, name_bez_nastavka)   
    return tocnost


data_folder_path='test-data'
dirs = os.listdir(data_folder_path) 

from datetime import datetime
start_time = datetime.now()  #sluzi da prikaz trajanja vremena


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
          tocnost= usporedbaImena(naziv, *predvideno_ime)
        #   print(tocnost)
          if(tocnost == 1): true_positive = true_positive +1
          else: true_negative=true_negative+1
         
print("Nasao sam tocno pozitivnih:", true_positive)
print("Nasao sam tocno negativnih:", true_negative)

end_time = datetime.now()
print('Vrijeme predvidanja: {}'.format(end_time - start_time))

br_dirs=len(dirs)-1 
print("Duzina dirs-a: ", br_dirs)
accuracy= (true_positive+true_negative)/br_dirs
accuracy_postotak=accuracy*100
print("accuracy: ", accuracy)
print("accuracy_posotak: ", accuracy_postotak, " % ")
