import imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import json
import pandas as pd
import seaborn as sns
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA



def get_img_volume(file_ext,blur=True):
    vid = imageio.get_reader(file_ext,  'ffmpeg')
    len_vid = vid.get_length()
    img_li = []
    for num in range(0,len_vid):
        image = vid.get_data(num)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if blur:
            image = cv2.GaussianBlur(image,(5,5),0)
        img_li.append(image)
    return np.array(img_li)


def compute_mhi(img_volume,threshold=30,show=False,starter=50):
    Z,Y,X = np.array(img_volume).shape
#    print X,Y,Z
#    print np.average(img_volume.flatten())
#    print np.std(img_volume.flatten())
    template = img_volume[0]
    movement_hist = []
    if show:
        plt.figure()
        plt.imshow(template)
        plt.show()
    for i in range(0,Z):
        diff = abs(template.astype(float)-img_volume[i].astype(float))
#        diff = np.where((diff>bright),0,diff)
        diff = np.where((diff<threshold),0,1)
        kernel = np.ones((2,2),np.uint8)
        erosion = cv2.erode((diff *1.0).astype(np.float32),kernel)
        if len(movement_hist) == 0:
            movement_hist=erosion*starter
        else:
            movement_hist+=erosion*starter
            movement_hist = np.where(movement_hist>starter,starter,movement_hist-1)
            movement_hist = np.where(movement_hist<0,0,movement_hist)
    if show:
        plt.figure()
        plt.imshow(movement_hist)
        plt.title('Movement History Image')
        plt.show()
    return movement_hist



def compute_mei(img_volume,threshold=30,show=False):
    Z,Y,X = np.array(img_volume).shape
    template = img_volume[0]
    if show:
        plt.figure()
        plt.imshow(template)
        plt.show()
    overall_diff = np.zeros([Y,X])
    for i in range(0,Z):
        diff = abs(template.astype(float)-img_volume[i].astype(float))
#        diff = np.where((diff>bright),0,diff)
        diff = np.where((diff<threshold),0,1)
        kernel = np.ones((2,2),np.uint8)
        erosion = cv2.erode((diff *1.0).astype(np.float32),kernel)
        overall_diff += erosion
    mei = np.where(overall_diff>0,1,0)
    if show:
        plt.figure()
        plt.imshow(overall_diff)
        plt.title('Overall Erosion Diff')
        plt.show()
        plt.figure()
        plt.title('Motion Energy Image')
        plt.imshow(mei)
        plt.show()
    return mei





def compute_hu_vals(motion_hist_img):
    Y,X = motion_hist_img.shape
    x_dir = np.array([range(1,X+1)]*Y)
    y_dir = np.array([range(1,Y+1)]*X).T

    M_0_0 = np.sum(motion_hist_img)
    M_0_1 = np.sum(y_dir*motion_hist_img)
    M_0_2 = np.sum(y_dir*y_dir*motion_hist_img)
    M_0_3 = np.sum(y_dir*y_dir*y_dir*motion_hist_img)

    M_1_0 = np.sum(x_dir*motion_hist_img)
    M_1_1 = np.sum(x_dir*y_dir*motion_hist_img)
    M_1_2 = np.sum(x_dir*y_dir*y_dir*motion_hist_img)

    M_2_0 = np.sum(x_dir*x_dir*motion_hist_img)
    M_2_1 = np.sum(x_dir*x_dir*y_dir*motion_hist_img)

    M_3_0 = np.sum(x_dir*x_dir*x_dir*motion_hist_img)

    x_bar = float(M_1_0)/M_0_0
    y_bar = float(M_0_1)/M_0_0

    mu_0_2 = M_0_2-(y_bar*M_0_1)
    mu_0_3 = M_0_3-(3*y_bar*M_0_2)+(2*(y_bar**2)*M_0_1)
    mu_1_1 = M_1_1-(x_bar*M_0_1)
    mu_1_2 = M_1_2-(2*y_bar*M_1_1)-(x_bar*M_0_2)+(2*(y_bar**2)*M_1_0)
    mu_2_0 = M_2_0-(x_bar*M_1_0)
    mu_2_1 = M_2_1-(2*x_bar*M_1_1)-(y_bar*M_2_0)+(2*(x_bar**2)*M_0_1)
    mu_3_0 = M_3_0-(3*x_bar*M_2_0)+(2*(x_bar**2)*M_1_0)
    
    h_1 = mu_2_0+mu_0_2
    h_2 = (mu_2_0+mu_0_2)**2 +4*(mu_1_1)**2
    h_3 = (mu_3_0 - 3*mu_1_2)**2 + (3*mu_2_1 - mu_0_3)**2
    h_4 = (mu_3_0 + mu_1_2) + (mu_2_1+mu_0_3)**2
    h_5 = ((mu_3_0 - 3*mu_1_2)*(mu_3_0 + mu_1_2)*((mu_3_0+mu_1_2)-3*(mu_2_1+mu_0_3)**2))+\
          ((3*mu_2_1 - mu_0_3)*(mu_2_1 + mu_0_3)*(3*(mu_3_0+mu_1_2)-(mu_2_1+mu_0_3)**2))
    h_6 = (mu_2_0-mu_0_2)*((mu_3_0+mu_1_2)**2 - (mu_2_1+mu_0_3)**2) + \
            4*mu_1_1*(mu_3_0+mu_1_2)*(mu_2_1+mu_0_3)
    h_7 = ((3*mu_2_1 - mu_0_3)*(mu_3_0 + mu_1_2)*((mu_3_0+mu_1_2)-3*(mu_2_1+mu_0_3)**2))-\
          ((mu_3_0-3*mu_1_2)*(mu_2_1+mu_0_3)*(3*(mu_3_0+mu_1_2)-(mu_2_1+mu_0_3)**2))
    
    return [h_1,h_2,h_3,h_4,h_5,h_6,h_7]

##################
#### 20-FRAME ####
##################


#Boxing, 20-frame depth
print 'Boxing'
path = 'input_files/boxing/'
hu_vals_boxing = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-20):
        mhi = compute_mhi(img_li[i:i+20],starter=25,threshold=30)
        hu_vals_boxing.append(compute_hu_vals(mhi))
sns.pairplot(pd.DataFrame(hu_vals_boxing))



#Handclapping, 20-frame depth
print 'Handclapping'
path = 'input_files/handclapping/'
hu_vals_clapping = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-20):
        mhi = compute_mhi(img_li[i:i+20],starter=25,threshold=30)
        hu_vals_clapping.append(compute_hu_vals(mhi))
sns.pairplot(pd.DataFrame(hu_vals_clapping))



#Handwaving, 20-frame depth
print 'Handwaving'
path = 'input_files/handwaving/'
hu_vals_handwaving = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-20):
        mhi = compute_mhi(img_li[i:i+20],starter=25,threshold=30)
        hu_vals_handwaving.append(compute_hu_vals(mhi))
sns.pairplot(pd.DataFrame(hu_vals_handwaving))



#Jogging, 20-frame depth
print 'Jogging'
path = 'input_files/jogging/'
hu_vals_jogging = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-20):
        mhi = compute_mhi(img_li[i:i+20],starter=25,threshold=30)
        hu_vals_jogging.append(compute_hu_vals(mhi))
sns.pairplot(pd.DataFrame(hu_vals_jogging))



#Running, 20-frame depth
print 'Running'
path = 'input_files/running/'
hu_vals_running = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-20):
        mhi = compute_mhi(img_li[i:i+20],starter=25,threshold=30)
        hu_vals_running.append(compute_hu_vals(mhi))
#sns.pairplot(pd.DataFrame(hu_vals_running))
        
        
        
#Walking, 20-frame depth
print 'Walking'
path = 'input_files/walking/'
hu_vals_walking = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-20):
        mhi = compute_mhi(img_li[i:i+20],starter=25,threshold=30)
        hu_vals_walking.append(compute_hu_vals(mhi))
#sns.pairplot(pd.DataFrame(hu_vals_walking))
        
#Save it all to a json object!
json_object_20 = {'boxing':hu_vals_boxing,\
                  'handclapping':hu_vals_clapping,\
                  'handwaving':hu_vals_handwaving,\
                  'jogging':hu_vals_jogging,\
                  'running':hu_vals_running,\
                  'walking':hu_vals_walking}
with open('hu_vals_20.json', 'w') as outfile:
    json.dump(json_object_20, outfile)
    
    
    
    
##################
#### 40-FRAME ####
##################

#Boxing, 40-frame depth
print 'Boxing'
path = 'input_files/boxing/'
hu_vals_boxing_40 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-40):
        mhi = compute_mhi(img_li[i:i+40],starter=45,threshold=30)
        hu_vals_boxing_40.append(compute_hu_vals(mhi))
sns.pairplot(pd.DataFrame(hu_vals_boxing_40))



#Handclapping, 40-frame depth
print 'Handclapping'
path = 'input_files/handclapping/'
hu_vals_clapping_40 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-40):
        mhi = compute_mhi(img_li[i:i+40],starter=45,threshold=30)
        hu_vals_clapping_40.append(compute_hu_vals(mhi))
sns.pairplot(pd.DataFrame(hu_vals_clapping_40))


#Handwaving, 40-frame depth
print 'Handwaving'
path = 'input_files/handwaving/'
hu_vals_handwaving_40 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-40):
        mhi = compute_mhi(img_li[i:i+40],starter=45,threshold=30)
        hu_vals_handwaving_40.append(compute_hu_vals(mhi))
sns.pairplot(pd.DataFrame(hu_vals_handwaving_40))


#Jogging, 40-frame depth
print 'Jogging'
path = 'input_files/jogging/'
hu_vals_jogging_40 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-40):
        mhi = compute_mhi(img_li[i:i+40],starter=45,threshold=30)
        hu_vals_jogging_40.append(compute_hu_vals(mhi))
#sns.pairplot(pd.DataFrame(hu_vals_jogging_40))


#Running, 40-frame depth
print 'Running'
path = 'input_files/running/'
hu_vals_running_40 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-40):
        mhi = compute_mhi(img_li[i:i+40],starter=45,threshold=30)
        hu_vals_running_40.append(compute_hu_vals(mhi))
#sns.pairplot(pd.DataFrame(hu_vals_running_40))


#Walking, 40-frame depth
print 'Walking'
path = 'input_files/walking/'
hu_vals_walking_40 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-40):
        mhi = compute_mhi(img_li[i:i+40],starter=45,threshold=30)
        hu_vals_walking_40.append(compute_hu_vals(mhi))
#sns.pairplot(pd.DataFrame(hu_vals_walking_40))


#Save all of it to a json object
json_object_40 = {'boxing':hu_vals_boxing_40,\
                  'handclapping':hu_vals_clapping_40,\
                  'handwaving':hu_vals_handwaving_40,\
                  'jogging':hu_vals_jogging_40,\
                  'running':hu_vals_running_40,\
                  'walking':hu_vals_walking_40}
with open('hu_vals_40.json', 'w') as outfile:
    json.dump(json_object_40, outfile)



###################
#### 100-FRAME ####
###################

#Boxing, 100-frame depth
print 'Boxing'
path = 'input_files/boxing/'
hu_vals_boxing_100 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-100):
        mhi = compute_mhi(img_li[i:i+100],starter=120,threshold=30)
        hu_vals_boxing_100.append(compute_hu_vals(mhi))
sns.pairplot(pd.DataFrame(hu_vals_boxing_100))


#Handclapping, 100-frame depth
print 'Handclapping'
path = 'input_files/handclapping/'
hu_vals_clapping_100 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-100):
        mhi = compute_mhi(img_li[i:i+100],starter=120,threshold=30)
        hu_vals_clapping_100.append(compute_hu_vals(mhi))
sns.pairplot(pd.DataFrame(hu_vals_clapping_100))


#Handwaving, 100-frame depth
print 'Handwaving'
path = 'input_files/handwaving/'
hu_vals_handwaving_100 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-100):
        mhi = compute_mhi(img_li[i:i+100],starter=120,threshold=30)
        hu_vals_handwaving_100.append(compute_hu_vals(mhi))
sns.pairplot(pd.DataFrame(hu_vals_handwaving_100))


#Jogging, 100-frame depth
print 'Jogging'
path = 'input_files/jogging/'
hu_vals_jogging_100 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-100):
        mhi = compute_mhi(img_li[i:i+100],starter=120,threshold=30)
        hu_vals_jogging_100.append(compute_hu_vals(mhi))
#sns.pairplot(pd.DataFrame(hu_vals_jogging_100))


#Running, 100-frame depth
print 'Running'
path = 'input_files/running/'
hu_vals_running_100 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-100):
        mhi = compute_mhi(img_li[i:i+100],starter=120,threshold=30)
        hu_vals_running_100.append(compute_hu_vals(mhi))
#sns.pairplot(pd.DataFrame(hu_vals_running_100))


#Walking, 100-frame depth
print 'Walking'
path = 'input_files/walking/'
hu_vals_walking_100 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-100):
        mhi = compute_mhi(img_li[i:i+100],starter=120,threshold=30)
        hu_vals_walking_100.append(compute_hu_vals(mhi))
#sns.pairplot(pd.DataFrame(hu_vals_walking_100))

json_object_100 = {'boxing':hu_vals_boxing_100,\
                  'handclapping':hu_vals_clapping_100,\
                  'handwaving':hu_vals_handwaving_100,\
                  'jogging':hu_vals_jogging_100,\
                  'running':hu_vals_running_100,\
                  'walking':hu_vals_walking_100}
with open('hu_vals_100.json', 'w') as outfile:
    json.dump(json_object_100, outfile)
    
    

###################
#### 250-FRAME ####
###################


#Boxing, 250-frame depth
print 'Boxing'
path = 'input_files/boxing/'
hu_vals_boxing_250 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-250):
        mhi = compute_mhi(img_li[i:i+250],starter=255,threshold=30)
        hu_vals_boxing_250.append(compute_hu_vals(mhi))
sns.pairplot(pd.DataFrame(hu_vals_boxing_250))


#Handclapping, 250-frame depth
print 'Handclapping'
path = 'input_files/handclapping/'
hu_vals_clapping_250 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-250):
        mhi = compute_mhi(img_li[i:i+250],starter=255,threshold=30)
        hu_vals_clapping_250.append(compute_hu_vals(mhi))
sns.pairplot(pd.DataFrame(hu_vals_clapping_250))


#Handwaving, 250-frame depth
print 'Handwaving'
path = 'input_files/handwaving/'
hu_vals_handwaving_250 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-250):
        mhi = compute_mhi(img_li[i:i+250],starter=255,threshold=30)
        hu_vals_handwaving_250.append(compute_hu_vals(mhi))
sns.pairplot(pd.DataFrame(hu_vals_handwaving_250))


#Jogging, 250-frame depth
print 'Jogging'
path = 'input_files/jogging/'
hu_vals_jogging_250 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-250):
        mhi = compute_mhi(img_li[i:i+250],starter=255,threshold=30)
        hu_vals_jogging_250.append(compute_hu_vals(mhi))
#sns.pairplot(pd.DataFrame(hu_vals_jogging_250))
        
        
#Running, 250-frame depth
print 'Running'
path = 'input_files/running/'
hu_vals_running_250 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-250):
        mhi = compute_mhi(img_li[i:i+250],starter=255,threshold=30)
        hu_vals_running_250.append(compute_hu_vals(mhi))
#sns.pairplot(pd.DataFrame(hu_vals_running_250))
        
        
        
        
#Walking, 250-frame depth
print 'Walking'
path = 'input_files/walking/'
hu_vals_walking_250 = []
for f in os.listdir(path):
    print f
    filename = os.path.join(path,f)
#    img_li = get_img_volume(filename,blur=True)
    img_li = get_img_volume(filename)
#    mei = compute_mei(img_li)
    for i in range(0,len(img_li)-250):
        mhi = compute_mhi(img_li[i:i+250],starter=255,threshold=30)
        hu_vals_walking_250.append(compute_hu_vals(mhi))
#sns.pairplot(pd.DataFrame(hu_vals_walking_250))
        
        
        
###########################################################
###########################################################
###########################################################
        
#Now we will do a little data analysis of our Hu moments.
        
        

with open('hu_vals_20_mhi.json') as json_data:
    hu_vals_20 = json.load(json_data)
    json_data.close()
print hu_vals_20.keys()

with open('hu_vals_40_mhi.json') as json_data:
    hu_vals_40 = json.load(json_data)
    json_data.close()
print hu_vals_40.keys()

with open('hu_vals_100_mhi.json') as json_data:
    hu_vals_100 = json.load(json_data)
    json_data.close()
print hu_vals_100.keys()

with open('hu_vals_250_mhi.json') as json_data:
    hu_vals_250 = json.load(json_data)
    json_data.close()
print hu_vals_250.keys()



for hu in range(0,7):
    plt.figure(figsize=(10,4))
    for i in hu_vals_20.keys():
        hu_li = np.array([j[hu] for j in hu_vals_20[i]])
        hu_li = hu_li[~np.isnan(hu_li)]
        mean = np.average(hu_li)
        standard_dev = np.std(hu_li)
        hu_li = [j for j in hu_li if (mean-standard_dev) < j]
        hu_li = [j for j in hu_li if j > (mean+standard_dev)]
        plt.hist(hu_li,50,alpha=0.6,label=i)
    plt.legend()
    plt.show()


for hu in range(0,6):
    plt.figure(figsize=(10,4))
    for i in hu_vals_40.keys():
        hu_li = np.array([j[hu] for j in hu_vals_40[i]])
        hu_li = hu_li[~np.isnan(hu_li)]
        mean = np.average(hu_li)
        standard_dev = np.std(hu_li)
        hu_li = [j for j in hu_li if (mean-standard_dev) < j]
        hu_li = [j for j in hu_li if j > (mean+standard_dev)]
        plt.hist(hu_li,50,alpha=0.6,label=i)
    plt.legend()
    plt.show()


for hu in range(0,6):
    plt.figure(figsize=(10,4))
    for i in hu_vals_100.keys():
        hu_li = np.array([j[hu] for j in hu_vals_100[i]])
        hu_li = hu_li[~np.isnan(hu_li)]
        mean = np.average(hu_li)
        standard_dev = np.std(hu_li)
        hu_li = [j for j in hu_li if (mean-standard_dev) < j]
        hu_li = [j for j in hu_li if j > (mean+standard_dev)]
        plt.hist(hu_li,50,alpha=0.6,label=i)
    plt.legend()
    plt.show()
    
    
    
for hu in range(0,6):
    plt.figure(figsize=(10,4))
    for i in hu_vals_250.keys():
        hu_li = np.array([j[hu] for j in hu_vals_250[i]])
        hu_li = hu_li[~np.isnan(hu_li)]
        mean = np.average(hu_li)
        standard_dev = np.std(hu_li)
        hu_li = [j for j in hu_li if (mean-standard_dev) < j]
        hu_li = [j for j in hu_li if j > (mean+standard_dev)]
        plt.hist(hu_li,50,alpha=0.6,label=i)
    plt.legend()
    plt.show()

##################### PCA and ICA #####################

overall_hu_20 = []
for i in hu_vals_20.keys():
    hu_vals_df = pd.DataFrame(hu_vals_20[i])
    hu_vals_df.dropna
    for j in hu_vals_df.values.tolist():
        overall_hu_20.append(j)
overall_hu_20 = np.array(overall_hu_20).astype('int')*float(10**-12)
print overall_hu_20[0]
#normalized_40 = normalize(overall_hu_40,norm='l2',axis=0)

PCA_apply_20 = PCA(n_components=2)
PCA_apply_20.fit(np.array(overall_hu_20))


plt.figure(figsize=(10,10))
PCA_computed = []
for key in hu_vals_20.keys():
    data = np.array(hu_vals_20[key]).astype('int')*float(10**-12)
    out_PCA = PCA_apply_20.transform(data)
    plt.scatter([i[0] for i in out_PCA],[i[1] for i in out_PCA],label=key,alpha=0.4)
    PCA_computed.append(out_PCA)
for dat in PCA_computed:
    dat = np.array(dat).T
    plt.scatter(np.average(dat[0]),np.average(dat[1]),color='red',s=250,marker='*')
plt.legend()
plt.show()


ICA_apply_20 = FastICA(n_components=2)
ICA_apply_20.fit(np.array(overall_hu_20))


plt.figure(figsize=(10,10))
ICA_computed = []
for key in hu_vals_20.keys():
    data = np.array(hu_vals_20[key]).astype('int')*float(10**-12)
    out_ICA = ICA_apply_20.transform(data)
    plt.scatter([i[0] for i in out_ICA],[i[1] for i in out_ICA],label=key,alpha=0.4)
    ICA_computed.append(out_ICA)
for dat in ICA_computed:
    dat = np.array(dat).T
    plt.scatter(np.average(dat[0]),np.average(dat[1]),color='red',s=250,marker='*')
plt.legend()
plt.show()











overall_hu_40 = []
for i in hu_vals_40.keys():
    hu_vals_df = pd.DataFrame(hu_vals_40[i])
    hu_vals_df.dropna
    for j in hu_vals_df.values.tolist():
        overall_hu_40.append(j)
overall_hu_40 = np.array(overall_hu_40).astype('int')*float(10**-12)
print overall_hu_40[0]
#normalized_40 = normalize(overall_hu_40,norm='l2',axis=0)


PCA_apply_40 = PCA(n_components=2)
PCA_apply_40.fit(np.array(overall_hu_40))


plt.figure(figsize=(10,10))
PCA_computed = []
for key in hu_vals_40.keys():
    data = np.array(hu_vals_40[key]).astype('int')*float(10**-12)
    out_PCA = PCA_apply_40.transform(data)
    plt.scatter([i[0] for i in out_PCA],[i[1] for i in out_PCA],label=key,alpha=0.4)
    PCA_computed.append(out_PCA)
for dat in PCA_computed:
    dat = np.array(dat).T
    plt.scatter(np.average(dat[0]),np.average(dat[1]),color='red',s=250,marker='*')
plt.legend()
plt.show()



ICA_apply_40 = FastICA(n_components=2)
ICA_apply_40.fit(np.array(overall_hu_40))


plt.figure(figsize=(10,10))
ICA_computed = []
for key in hu_vals_40.keys():
    data = np.array(hu_vals_40[key]).astype('int')*float(10**-12)
    out_ICA = ICA_apply_40.transform(data)
    plt.scatter([i[0] for i in out_ICA],[i[1] for i in out_ICA],label=key,alpha=0.4)
    ICA_computed.append(out_ICA)
for dat in ICA_computed:
    dat = np.array(dat).T
    plt.scatter(np.average(dat[0]),np.average(dat[1]),color='red',s=250,marker='*')
plt.legend()
plt.show()










overall_hu_100 = []
for i in hu_vals_100.keys():
    hu_vals_df = pd.DataFrame(hu_vals_100[i])
    hu_vals_df.dropna
    for j in hu_vals_df.values.tolist():
        overall_hu_100.append(j)
overall_hu_100 = np.array(overall_hu_100).astype('int')*float(10**-12)
print overall_hu_100[0]
#normalized_40 = normalize(overall_hu_40,norm='l2',axis=0)

PCA_apply_100 = PCA(n_components=2)
PCA_apply_100.fit(np.array(overall_hu_100))


plt.figure(figsize=(10,10))
PCA_computed = []
for key in hu_vals_100.keys():
    data = np.array(hu_vals_100[key]).astype('int')*float(10**-12)
    out_PCA = PCA_apply_100.transform(data)
    plt.scatter([i[0] for i in out_PCA],[i[1] for i in out_PCA],label=key,alpha=0.4)
    PCA_computed.append(out_PCA)
for dat in PCA_computed:
    dat = np.array(dat).T
    plt.scatter(np.average(dat[0]),np.average(dat[1]),color='red',s=250,marker='*')
plt.legend()
plt.show()


ICA_apply_100 = FastICA(n_components=2)
ICA_apply_100.fit(np.array(overall_hu_100))


plt.figure(figsize=(10,10))
ICA_computed = []
for key in hu_vals_100.keys():
    data = np.array(hu_vals_100[key]).astype('int')*float(10**-12)
    out_ICA = ICA_apply_100.transform(data)
    plt.scatter([i[0] for i in out_ICA],[i[1] for i in out_ICA],label=key,alpha=0.4)
    ICA_computed.append(out_ICA)
for dat in ICA_computed:
    dat = np.array(dat).T
    plt.scatter(np.average(dat[0]),np.average(dat[1]),color='red',s=250,marker='*')
plt.legend()
plt.show()










overall_hu_250 = []
for i in hu_vals_250.keys():
    hu_vals_df = pd.DataFrame(hu_vals_250[i])
    hu_vals_df.dropna
    for j in hu_vals_df.values.tolist():
        overall_hu_250.append(j)
overall_hu_250 = np.array(overall_hu_250).astype('int')*float(10**-12)
print overall_hu_250[0]
#normalized_40 = normalize(overall_hu_40,norm='l2',axis=0)

PCA_apply_250 = PCA(n_components=2)
PCA_apply_250.fit(np.array(overall_hu_250))

plt.figure(figsize=(10,10))
PCA_computed = []
for key in hu_vals_250.keys():
    data = np.array(hu_vals_250[key]).astype('int')*float(10**-12)
    out_PCA = PCA_apply_250.transform(data)
    plt.scatter([i[0] for i in out_PCA],[i[1] for i in out_PCA],label=key,alpha=0.4)
    PCA_computed.append(out_PCA)
for dat in PCA_computed:
    dat = np.array(dat).T
    plt.scatter(np.average(dat[0]),np.average(dat[1]),color='red',s=250,marker='*')
plt.legend()
plt.show()


ICA_apply_250 = FastICA(n_components=2)
ICA_apply_250.fit(np.array(overall_hu_250))

plt.figure(figsize=(10,10))
ICA_computed = []
for key in hu_vals_250.keys():
    data = np.array(hu_vals_250[key]).astype('int')*float(10**-12)
    out_ICA = ICA_apply_250.transform(data)
    plt.scatter([i[0] for i in out_ICA],[i[1] for i in out_ICA],label=key,alpha=0.4)
    ICA_computed.append(out_ICA)
for dat in ICA_computed:
    dat = np.array(dat).T
    plt.scatter(np.average(dat[0]),np.average(dat[1]),color='red',s=250,marker='*')
plt.legend()
plt.show()