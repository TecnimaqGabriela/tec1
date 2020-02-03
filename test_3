import json
import numpy as np
import skimage
from skimage import io
import matplotlib.pyplot as plt 
import time

with open('/home/tecnimaq/Gabriela/TF-SimpleHumanPose/output/result/entrenamiento_5/result.json') as openjson_5:
    entr_5 = json.load(openjson_5)

with open('/home/tecnimaq/Gabriela/TF-SimpleHumanPose/output/result/entrenamiento_5v2/result.json') as openjson_5v2:
    entr_5v2 = json.load(openjson_5v2)

with open('/home/tecnimaq/Gabriela/TF-SimpleHumanPose/output/result/entrenamiento_5v6/result.json') as openjson_5v6:
    entr_5v6 = json.load(openjson_5v6)

with open('/home/tecnimaq/Gabriela/TF-SimpleHumanPose/output/result/entrenamiento_5v3/result.json') as openjson_5v3:
    entr_5v3 = json.load(openjson_5v3)

with open('/home/tecnimaq/Gabriela/TF-SimpleHumanPose/output/result/entrenamiento_6v2/result.json') as openjson_6v2:
    entr_6v2 = json.load(openjson_6v2)

lote_size = 4

dimension = 6 # 1 triangle with 3 points (shoulder, hip and knee), 2 coordinates each: 2x3=6
amount = int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3)+len(entr_6v2))/lote_size)-7
amount_in5 = int(len(entr_5)/lote_size)
keypoints_dimension = len(entr_5[0].get('keypoints'))
data = np.zeros((amount, lote_size, dimension), dtype = float)
test_data = np.zeros((amount, lote_size, dimension), dtype = float)
image_names = np.zeros((amount, lote_size), dtype = int)
dataset_name = np.zeros(amount, dtype = float)
keypoints = np.zeros((amount, lote_size, keypoints_dimension), dtype = float)

for lote in range(amount):

    for obj in range(lote_size):

        if lote < int(len(entr_5)/lote_size):

            keypoints[lote,obj] = entr_5[obj+(lote*lote_size)].get('keypoints')
            image_names[lote,obj] = entr_5[obj+(lote*lote_size)].get('image_id')
            dataset_name[lote] = 5

        elif int(len(entr_5)/lote_size)-1 < lote < int((len(entr_5)+len(entr_5v2))/lote_size):

            keypoints[lote,obj] = entr_5v2[obj+(lote-(int(len(entr_5)/lote_size)))*lote_size].get('keypoints')
            image_names[lote,obj] = entr_5v2[obj+(lote-(int(len(entr_5)/lote_size)))*lote_size].get('image_id')
            dataset_name[lote] = 5.2
    
        elif int((len(entr_5)+len(entr_5v2))/lote_size)-1 < lote < int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size):

            keypoints[lote,obj] = entr_5v6[obj+(lote-int((len(entr_5)+len(entr_5v2))/lote_size))*lote_size].get('keypoints')
            image_names[lote,obj] = entr_5v6[obj+(lote-int((len(entr_5)+len(entr_5v2))/lote_size))*lote_size].get('image_id')
            dataset_name[lote] = 5.6
        
        elif int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size)-1 < lote < int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)-1:

            keypoints[lote,obj] = entr_5v3[obj+(lote-int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size))*lote_size].get('keypoints')
            image_names[lote,obj] = entr_5v3[obj+(lote-int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size))*lote_size].get('image_id')
            dataset_name[lote] = 5.3

        else:

            keypoints[lote,obj] = entr_6v2[obj+(lote+1-int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size))*lote_size].get('keypoints')
            image_names[lote,obj] = entr_6v2[obj+(lote+1-int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size))*lote_size].get('image_id')
            dataset_name[lote] = 6.2

        data[lote][obj][0] = keypoints[lote][obj][15] - keypoints[lote][obj][39]
        data[lote][obj][1] = 0
        data[lote][obj][2] = keypoints[lote][obj][33] - keypoints[lote][obj][39]
        data[lote][obj][3] = keypoints[lote][obj][34] - keypoints[lote][obj][16]
        data[lote][obj][4] = 0
        data[lote][obj][5] = keypoints[lote][obj][40] - keypoints[lote][obj][16]

with open('/home/tecnimaq/Gabriela/TF-SimpleHumanPose/data/entrenamiento_5/labelsx4_only5.json') as openjson_lab:
    labels = json.load(openjson_lab)


label1 = np.zeros((amount_in5,1), dtype = int)
label2 = np.zeros((amount_in5,1), dtype = int)
dataset = np.zeros(amount_in5, dtype = int)
estado = np.zeros(amount_in5, dtype = int)
images_lote = np.zeros(amount_in5, dtype = int)

for i in range(amount_in5):
    for m in range(amount_in5):
        if m == labels[i].get('obj'):
            label1[m][0] = labels[i].get('label')
            label2[m][0] = labels[i].get('lab_2')
            estado[m] = labels[i].get('estado')
            dataset[m] = labels[i].get('dataset')
            images_lote[m] = labels[i].get('id')

Sentado = np.zeros((50,lote_size,dimension), dtype = float)
Images_Sentado = np.zeros((50,lote_size), dtype = int)
Dataset_Sentado = np.zeros(50, dtype = float)
n = 0
for lote in range(amount_in5):
    if n < 50:
        if estado[lote] == 1:
            Sentado[n] = data[lote]
            # print('Data with estado = 1: ', data[lote])
            Images_Sentado[n] = image_names[lote]
            # print(n,'th data in Sentado: ', Sentado[n])
            # print('from images: ', image_names[n])
            Dataset_Sentado[n] = dataset_name[lote]
            n = n+1
            
max_in_sentado = n

De_pie = np.zeros((50,lote_size,dimension), dtype = float)
Images_Depie = np.zeros((50,lote_size), dtype = int)
Dataset_Depie = np.zeros(50, dtype = float)
n = 0
for lote in range(amount_in5):
    if n < 50:
        if estado[lote] == 2:
            De_pie[n] = data[lote]
            Images_Depie[n] = image_names[lote]
            Dataset_Depie[n] = dataset_name[lote]
            n = n+1
        
max_in_depie = n

Sentandose = np.zeros((50,lote_size,dimension), dtype = float)
Images_Sentandose = np.zeros((50,lote_size), dtype = int)
Dataset_Sentandose = np.zeros(50, dtype = float)
n = 0
for lote in range(amount_in5):
    if n < 50:
        if label2[lote] == 2:
            Sentandose[n] = data[lote]
            Images_Sentandose[n] = image_names[lote]
            Dataset_Sentandose[n] = dataset_name[lote]
            n = n+1

max_n_insendandose = n

Levantandose = np.zeros((50,lote_size,dimension), dtype = float)
Images_Levantandose = np.zeros((50,lote_size), dtype = int)
Dataset_Levantandose = np.zeros(50, dtype = float)
n = 0
for lote in range(amount_in5):
    if n < 50:
        if label2[lote] == 1:
            Levantandose[n] = data[lote]
            Images_Levantandose[n] = image_names[lote]
            Dataset_Levantandose[n] = dataset_name[lote]
            n = n+1
       
max_n_inlevantandose = n

with open('/home/tecnimaq/Gabriela/TF-SimpleHumanPose/data/entrenamiento_5/labelsx4_only52.json') as openjson_lab:
    labels_2 = json.load(openjson_lab)

amount_in52 = int(len(entr_5v2)/lote_size)

label1_2 = np.zeros((amount_in52,1), dtype = int)
label2_2 = np.zeros((amount_in52,1), dtype = int)
dataset_2 = np.zeros(amount_in52, dtype = int)
estado_2 = np.zeros(amount_in52, dtype = int)
images_lote_2 = np.zeros(amount_in52, dtype = int)

for i in range(amount_in52):
    for m in range(amount_in52):
        if m+int(len(entr_5)/lote_size) == labels_2[i].get('obj'):
            label1_2[m][0] = labels_2[i].get('label')
            label2_2[m][0] = labels_2[i].get('lab_2')
            # estado[m] = labels[i].get('estado')
            dataset_2[m] = labels_2[i].get('dataset')
            images_lote_2[m] = labels_2[i].get('id')
    
n = max_n_insendandose
for lote in range(amount_in52):
    if n < 50:
        if label2_2[lote] == 2:
            Sentandose[n] = data[lote+int(len(entr_5)/lote_size)]
            Images_Sentandose[n] = image_names[lote+int(len(entr_5)/lote_size)]
            Dataset_Sentandose[n] = dataset_name[lote+int(len(entr_5)/lote_size)]
            n = n+1
max_n_insendandose = n

n = max_n_inlevantandose
for lote in range(amount_in52):
    if n < 50:
        if label2_2[lote] == 1:
            Levantandose[n] = data[lote+int(len(entr_5)/lote_size)]
            Images_Levantandose[n] = image_names[lote+int(len(entr_5)/lote_size)]
            Dataset_Levantandose[n] = dataset_name[lote+int(len(entr_5)/lote_size)]
            n = n+1
max_n_inlevantandose = n

with open('/home/tecnimaq/Gabriela/TF-SimpleHumanPose/data/entrenamiento_5/labelsx4_only56.json') as openjson_lab:
    labels_6 = json.load(openjson_lab)

amount_in56 = int(len(entr_5v6)/lote_size)

label1_6 = np.zeros((amount_in56,1), dtype = int)
label2_6 = np.zeros((amount_in56,1), dtype = int)
dataset_6 = np.zeros(amount_in56, dtype = int)
estado_6 = np.zeros(amount_in56, dtype = int)
images_lote_6 = np.zeros(amount_in56, dtype = int)

for i in range(amount_in56):
    for m in range(amount_in56):
        if m+int((len(entr_5)+len(entr_5v2))/lote_size) == labels_6[i].get('obj'):
            label1_6[m][0] = labels_6[i].get('label')
            label2_6[m][0] = labels_6[i].get('lab_2')
            # estado[m] = labels[i].get('estado')
            dataset_6[m] = labels_6[i].get('dataset')
            images_lote_6[m] = labels_6[i].get('id')
    
n = max_n_insendandose
for lote in range(amount_in56):
    if n < 50:
        if label2_6[lote] == 2:
            Sentandose[n] = data[lote+int((len(entr_5)+len(entr_5v2))/lote_size)]
            Images_Sentandose[n] = image_names[lote+int((len(entr_5)+len(entr_5v2))/lote_size)]
            Dataset_Sentandose[n] = dataset_name[lote+int((len(entr_5)+len(entr_5v2))/lote_size)]
            n = n+1

max_n_insentandose = n

n = max_n_inlevantandose
for lote in range(amount_in56):
    if n < 50:
        if label2_6[lote] == 1:
            Levantandose[n] = data[lote+int((len(entr_5)+len(entr_5v2))/lote_size)]
            Images_Levantandose[n] = image_names[lote+int((len(entr_5)+len(entr_5v2))/lote_size)]
            Dataset_Levantandose[n] = dataset_name[lote+int((len(entr_5)+len(entr_5v2))/lote_size)]
            n = n+1

max_n_inlevantandose = n
 
with open('/home/tecnimaq/Gabriela/TF-SimpleHumanPose/data/entrenamiento_5/labelsx4_only53.json') as openjson_lab:
    labels_3 = json.load(openjson_lab)

amount_in53 = int(len(entr_5v3)/lote_size)

label1_3 = np.zeros((amount_in53,1), dtype = int)
label2_3 = np.zeros((amount_in53,1), dtype = int)
dataset_3 = np.zeros(amount_in53, dtype = int)
estado_3 = np.zeros(amount_in53, dtype = int)
images_lote_3 = np.zeros(amount_in53, dtype = int)

for i in range(amount_in53):
    for m in range(amount_in53):
        if m+int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size) == labels_3[i].get('obj'):
            label1_3[m][0] = labels_3[i].get('label')
            label2_3[m][0] = labels_3[i].get('lab_2')
            estado_3[m] = labels_3[i].get('estado')
            dataset_3[m] = labels_3[i].get('dataset')
            images_lote_3[m] = labels_3[i].get('id')

n = max_in_sentado

for lote in range(amount_in53):
    if n < 50:
        if estado_3[lote] == 1:
            Sentado[n] = data[lote+int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size)]
            Images_Sentado[n] = image_names[lote+int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size)]
            Dataset_Sentado[n] = dataset_name[lote+int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size)]
            n = n+1

max_in_sentado = n

n = max_in_depie
for lote in range(amount_in53):
    if n < 25:
        if estado_3[lote] == 2:
            De_pie[n] = data[lote+int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size)]
            Images_Depie[n] = image_names[lote+int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size)]
            Dataset_Depie[n] = dataset_name[lote+int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size)]
            n = n+1
        
max_in_depie = n

n = max_n_insendandose
for lote in range(amount_in53):
    if n < 50:
        if label2_3[lote] == 2:
            Sentandose[n] = data[lote+int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size)]
            Images_Sentandose[n] = image_names[lote+int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size)]
            Dataset_Sentandose[n] = dataset_name[lote+int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size)]
            n = n+1

max_n_insendandose = n

n = max_n_inlevantandose
for lote in range(amount_in53):
    if n < 50:
        if label2_3[lote] == 1:
            Levantandose[n] = data[lote+int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size)]
            Images_Levantandose[n] = image_names[lote+int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size)]
            Dataset_Levantandose[n] = dataset_name[lote+int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size)]
            n = n+1
        
max_n_inlevantandose = n

with open('/home/tecnimaq/Gabriela/TF-SimpleHumanPose/data/entrenamiento_5/labelsx4_only62.json') as openjson_lab:
    labels_62 = json.load(openjson_lab)

amount_in62 = int(len(entr_6v2)/lote_size)

label1_62 = np.zeros((amount_in62,1), dtype = int)
label2_62 = np.zeros((amount_in62,1), dtype = int)
dataset_62 = np.zeros(amount_in62, dtype = int)
estado_62 = np.zeros(amount_in62, dtype = int)
images_lote_62 = np.zeros(amount_in62, dtype = int)

for i in range(amount_in62-7):
    for m in range(amount_in62-7):
        if m+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)-1 == labels_62[i].get('obj'):
            label1_62[m][0] = labels_62[i].get('label')
            label2_62[m][0] = labels_62[i].get('lab_2')
            estado_62[m] = labels_62[i].get('estado')
            dataset_62[m] = labels_62[i].get('dataset')
            images_lote_62[m] = labels_62[i].get('id')
            
n = 0
for lote in range(amount_in62):
    if n < 15:
        if estado_62[lote] == 1:
            Sentado[n+5] = data[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            Images_Sentado[n+5] = image_names[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            Dataset_Sentado[n+5] = dataset_name[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            n = n+1

n = max_in_depie
for lote in range(amount_in62):
    if n < 50:
        if estado_62[lote] == 2:
            De_pie[n] = data[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            Images_Depie[n] = image_names[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            Dataset_Depie[n] = dataset_name[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            n = n+1

n = max_n_insendandose
for lote in range(amount_in62):
    if n < 50:
        if label2_62[lote] == 2:
            Sentandose[n] = data[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            Images_Sentandose[n] = image_names[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            Dataset_Sentandose[n] = dataset_name[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            n = n+1

max_n_insendandose = n

n = max_n_insendandose
for lote in range(amount_in62):
    if n < 50:
        if label2_62[lote] == 2:
            Sentandose[n] = data[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            Images_Sentandose[n] = image_names[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            Dataset_Sentandose[n] = dataset_name[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            n = n+1

max_n_insendandose = n

n = max_n_insendandose
for lote in range(amount_in62):
    if n < 50:
        if label2_62[lote] == 2:
            Sentandose[n] = data[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            Images_Sentandose[n] = image_names[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            Dataset_Sentandose[n] = dataset_name[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            n = n+1

max_n_insendandose = n

n = max_n_inlevantandose
for lote in range(amount_in62):
    if n < 50:
        if label2_62[lote] == 1:
            Levantandose[n] = data[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            Images_Levantandose[n] = image_names[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            Dataset_Levantandose[n] = dataset_name[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            n = n+1
        
max_n_inlevantandose = n

n = max_n_inlevantandose
for lote in range(amount_in62):
    if n < 50:
        if label2_62[lote] == 1:
            Levantandose[n] = data[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            Images_Levantandose[n] = image_names[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            Dataset_Levantandose[n] = dataset_name[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            n = n+1
        
max_n_inlevantandose = n

n = max_n_inlevantandose
for lote in range(amount_in53):
    if n < 50:
        if label2_3[lote] == 1:
            Levantandose[n] = data[lote+int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size)]
            Images_Levantandose[n] = image_names[lote+int((len(entr_5)+len(entr_5v2)+len(entr_5v6))/lote_size)]
            Dataset_Levantandose[n] = dataset_name[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            n = n+1
        
max_n_inlevantandose = n

n = max_n_inlevantandose
for lote in range(amount_in56):
    if n < 50:
        if label2_6[lote] == 1:
            Levantandose[n] = data[lote+int((len(entr_5)+len(entr_5v2))/lote_size)]
            Images_Levantandose[n] = image_names[lote+int((len(entr_5)+len(entr_5v2))/lote_size)]
            Dataset_Levantandose[n] = dataset_name[lote-1+int((len(entr_5)+len(entr_5v2)+len(entr_5v6)+len(entr_5v3))/lote_size)]
            n = n+1

final_data = np.zeros((200, lote_size, dimension), dtype = float)
images_infinaldata = np.zeros((200, lote_size), dtype = int)
dataset_infinaldata = np.zeros(200, dtype = float)

for lote in range(200):
    for obj in range(lote_size):
        if lote < 50:
            final_data[lote][obj] = Sentado[lote][obj]
            images_infinaldata[lote] = Images_Sentado[lote]
            dataset_infinaldata[lote] = Dataset_Sentado[lote]
            # print(final_data[lote][obj])
            # print(lote,': Sentados Datasets: ', dataset_infinaldata[lote])
        elif 49 < lote < 100:
            final_data[lote][obj] = De_pie[lote-50][obj]
            images_infinaldata[lote] = Images_Depie[lote-50]
            dataset_infinaldata[lote] = Dataset_Depie[lote-50]
            # print(lote-50,': De pie Datasets: ', dataset_infinaldata[lote-50])
        elif 99 < lote < 150:
            final_data[lote][obj] = Sentandose[lote-100][obj]
            images_infinaldata[lote] = Images_Sentandose[lote-100]
            dataset_infinaldata[lote] = Dataset_Sentandose[lote-100]
            #print(lote-100,': Sendandose Dataset: ', dataset_infinaldata[lote], 'with images ', images_infinaldata[lote])
        else:
            final_data[lote][obj] = Levantandose[lote-150][obj]
            images_infinaldata[lote] = Images_Levantandose[lote-150]
            dataset_infinaldata[lote] = Dataset_Levantandose[lote-150]
            #print(lote-150,': Levantandose Dataset: ', dataset_infinaldata[lote], 'with images ', images_infinaldata[lote])

#print(dataset_infinaldata)
label = np.zeros((200,3), dtype = int)

for lote in range(200):
    if lote < 100:
        label[lote] = [0,0,1]
    elif 99 < lote < 150:
        label[lote] = [0,1,0] # sentandose
    else:
        label[lote] = [1,0,0] # levantandose

data_mean = final_data.mean()
normalized_data = (final_data - data_mean)

import sklearn
from sklearn.model_selection import train_test_split
    
X_train, X_test, y_train, y_test = train_test_split(normalized_data, label, test_size = 0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Flatten
from tensorflow.keras.callbacks import TensorBoard

NAME = 'Model_#{}'.format(int(time.time()))

tensorboard = TensorBoard(log_dir = 'logs/{}'.format(NAME))

model = tf.keras.Sequential()
model.add(LSTM(64, activation = 'tanh', input_shape =(4,6), return_sequences=True))
model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 20, validation_data = (X_val, y_val), callbacks=[tensorboard])

scores = model.evaluate(X_train, y_train)
print('Training Accuracy:{}%'.format(int(scores[1]*100)))
scores = model.evaluate(X_test, y_test)
print('Testing Accuracy:{}%'.format(int(scores[1]*100)))

with open('/home/tecnimaq/Gabriela/TF-SimpleHumanPose/output/result/val_5/result.json') as openjson_5:
    test_5 = json.load(openjson_5)

lote_size = 4

dimension = 6 # 2 triangles (left and right), with 3 points (shoulder, hip and knee), 2 coordinates each: 2x3x2=12
test_amount = int(len(test_5)/lote_size) # That is (200 + 400 + 400)/10 = 1000/10 = 100
keypoints_dimension = len(test_5[0].get('keypoints'))
test_data = np.zeros((test_amount, lote_size, dimension), dtype = float)
test_image_names = np.zeros((test_amount, lote_size), dtype = int)
test_keypoints = np.zeros((test_amount, lote_size, keypoints_dimension), dtype = float)

for lote in range(test_amount):

    for obj in range(lote_size):

        test_keypoints[lote,obj] = test_5[obj+(lote*lote_size)].get('keypoints')
        test_image_names[lote,obj] = test_5[obj+(lote*lote_size)].get('image_id')
        test_data[lote][obj][0] = test_keypoints[lote][obj][15] - test_keypoints[lote][obj][39]
        test_data[lote][obj][1] = 0
        test_data[lote][obj][2] = test_keypoints[lote][obj][33] - test_keypoints[lote][obj][39]
        test_data[lote][obj][3] = test_keypoints[lote][obj][34] - test_keypoints[lote][obj][16]
        test_data[lote][obj][4] = 0
        test_data[lote][obj][5] = test_keypoints[lote][obj][40] - test_keypoints[lote][obj][16]

testdata_mean = test_data.mean()
normalized_testdata = (test_data - testdata_mean)    
prediction = model.predict(normalized_testdata)

for lote in range(test_amount):
    for i in range(lote_size):
        if prediction[lote][1] > 0.7:
            path_image = '/home/tecnimaq/Gabriela/TF-SimpleHumanPose/data/val_5/images/test2017/'+str(test_image_names[lote][i-lote_size])+'.jpg'
            image = io.imread(path_image)
            plt.imshow(image)
            plt.scatter(test_data[lote][i-lote_size][0],test_data[lote][i-lote_size][1])
            plt.scatter(test_data[lote][i-lote_size][2],test_data[lote][i-lote_size][3])
            plt.scatter(test_data[lote][i-lote_size][4],test_data[lote][i-lote_size][5])
            plt.title('Sentandose: Lote {}, Image {}, Prediction {}'.format(lote, test_image_names[lote][i-lote_size], prediction[lote]))
            #plt.show()
        if prediction[lote][0] > 0.7:
            path_image = '/home/tecnimaq/Gabriela/TF-SimpleHumanPose/data/val_5/images/test2017/'+str(test_image_names[lote][i-lote_size])+'.jpg'
            image = io.imread(path_image)
            plt.imshow(image)
            plt.scatter(test_data[lote][i-lote_size][0],test_data[lote][i-lote_size][1])
            plt.scatter(test_data[lote][i-lote_size][2],test_data[lote][i-lote_size][3])
            plt.scatter(test_data[lote][i-lote_size][4],test_data[lote][i-lote_size][5])
            plt.title('Levantandose: Lote {}, Image {}, Prediction {}'.format(lote, test_image_names[lote][i-lote_size], prediction[lote]))
            #plt.show()

print(prediction)

