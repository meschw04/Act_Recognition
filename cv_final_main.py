import imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import json
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def get_img_volume(file_ext,blur=True):
    '''
    Gets the full volume of the image and returns it as an array.
    Args:
        file_ext = string, full file path
        blur = bool, is Gaussian blurring desired?
    Returns:
        3d array of layered images
    '''
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
    '''
    Computes the MHI (motion history image)
    Args:
        img_volume = np.array(), 3d volume of layered images
        threshold = int, difference in pixels to get recorded
        show = bool, use matplotlib to display the image (y/n)?
        starter = int, starting intensity of the MHI for recent motions
    Returns:
        2d array, computed MHI
    '''
    Z,Y,X = np.array(img_volume).shape
    template = img_volume[0]
    movement_hist = []
    if show:
        plt.figure()
        plt.imshow(template)
        plt.show()
    for i in range(0,Z):
        diff = abs(template.astype(float)-img_volume[i].astype(float))
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
    '''
    Computes the MEI (motion energy image)
    Args:
        img_volume = np.array(), 3d volume of layered imges
        threshold = int, difference in pixels to get recorded
        show = bool, use matplotlib to display the image (y/n)?
    Returns:
        2d array, computed MEI
    '''
    Z,Y,X = np.array(img_volume).shape
    template = img_volume[0]
    if show:
        plt.figure()
        plt.imshow(template)
        plt.show()
    overall_diff = np.zeros([Y,X])
    for i in range(0,Z):
        diff = abs(template.astype(float)-img_volume[i].astype(float))
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
    '''
    Computes the Hu values
    Args:
        motion_his_img = np.array(), the MHI
    Returns:
        List of the seven Hu values to be used.
    '''
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




def write_video(hu_info,knn_data,filename,true_state,out_video,show=True,write_video=True):
    '''
    Writes the video for the selected video file.
    Args:
        hu_info = np.array() Database selection of the Hu values for training data
        knn_data = List of objects, KNN models for finding nearest neighbors
        filename = Name of the video file read in to be computed on
        out_video = Name the outputted video will be called
        show = bool, use matplotlib to display the image (y/n)?
        write_video = bool, do you want the video to be saved?
    Returns:
        Final state representing what is believed to be the actual state of the system.
    '''
    overall_hu_20,overall_hu_40,overall_hu_100,overall_hu_250,\
    hu_20_labels, hu_40_labels, hu_100_labels, hu_250_labels = hu_info
    knn20_hu_abbrev,knn40_hu_abbrev,knn100_hu_abbrev,knn250_hu_abbrev = knn_data

    if write_video:
        writer = imageio.get_writer(out_video, fps=30)
    img_li = get_img_volume(filename)
    activities_li = ['boxing','handclapping','handwaving','jogging','running','walking']
    activities_count_20 = list(np.zeros(len(activities_li)))
    activities_count_40 = list(np.zeros(len(activities_li)))
    activities_count_100 = list(np.zeros(len(activities_li)))
    activities_count_250 = list(np.zeros(len(activities_li)))
    for i in range(0,len(img_li)):
        if i in range(0,len(img_li)-20):
            mhi = compute_mhi(img_li[i:i+20],starter=25,threshold=30)
            hu = compute_hu_vals(mhi)
            hu = [k*float(10**-12) for k in hu]
            try:
                ds_20, inds_20 = knn20_hu_abbrev.kneighbors([hu[:2]])
            except ValueError:
                continue #This defaults to the directly previous state
            act_sub_li = [hu_20_labels[q] for q in inds_20[0]]
            for act in act_sub_li[:3]:
                activities_count_20[activities_li.index(act)]+=1
        
        if i in range(0,len(img_li)-40):
            mhi = compute_mhi(img_li[i:i+40],starter=45,threshold=30)
            hu = compute_hu_vals(mhi)
            hu = [k*float(10**-12) for k in hu]
            try:
                ds_40, inds_40 = knn40_hu_abbrev.kneighbors([hu[:2]])
            except ValueError:
                continue #This defaults to the directly previous state
            act_sub_li = [hu_40_labels[q] for q in inds_40[0]]
            for act in act_sub_li[:3]:
                activities_count_40[activities_li.index(act)]+=1
        
        if i in range(0,len(img_li)-100):
            mhi = compute_mhi(img_li[i:i+100],starter=120,threshold=30)
            hu = compute_hu_vals(mhi)
            hu = [k*float(10**-12) for k in hu]
            try:
                ds_100, inds_100 = knn100_hu_abbrev.kneighbors([hu[:2]])
            except ValueError:
                continue #This defaults to the directly previous state
            act_sub_li = [hu_100_labels[q] for q in inds_100[0]]
            for act in act_sub_li[:3]:
                activities_count_100[activities_li.index(act)]+=1
        
        if i in range(0,len(img_li)-250):
            mhi = compute_mhi(img_li[i:i+250],starter=255,threshold=30)
            hu = compute_hu_vals(mhi)
            hu = [k*float(10**-12) for k in hu]
            try:
                ds_250, inds_250 = knn250_hu_abbrev.kneighbors([hu[:2]])
            except ValueError:
                continue #This defaults to the directly previous state
            act_sub_li = [hu_250_labels[q] for q in inds_250[0]]
            for act in act_sub_li[:3]:
                activities_count_250[activities_li.index(act)]+=1

        activities_count = list(np.array(activities_count_20)+np.array(activities_count_40)+\
                           np.array(activities_count_100)+np.array(activities_count_250))
        final_set_state = activities_li[activities_count.index(max(activities_count))]

        if write_video:
            plt.figure(figsize=(10,10))
            #Original Frame
            plt.subplot(2, 2, 1)
            plt.title('Image Frame '+str(i))
            plt.imshow(np.array(img_li[i]),cmap='gray')
            plt.text(5,110,'Best State: '+str(final_set_state),color='red',fontsize=16)
            plt.text(5,10,'True Answer: '+str(true_state),color='green',fontsize=16)
            #Motion Energy Image
            if i>0:
                plt.subplot(2,2,2)
                plt.title('Motion Energy Image')
                plt.imshow(compute_mei(img_li[:i]))
                #Motion History Image
                plt.subplot(2,2,3)
                plt.title('Motion History Image')
                plt.imshow(compute_mhi(img_li[:i],starter=255,threshold=30))
            else:
                plt.subplot(2,2,2)
                plt.title('Motion Energy Image')
                plt.imshow(np.zeros(np.array(img_li[0]).shape))
                plt.subplot(2,2,3)
                plt.title('Motion History Image')
                plt.imshow(np.zeros(np.array(img_li[0]).shape))
            #3-Nearest Count
            plt.subplot(2,2,4)
            bar_x = np.linspace(0.5,5.5,6)
            bar_x_20 = np.linspace(0.125,5.125,6)
            bar_x_40 = np.linspace(0.375,5.375,6)
            bar_x_100 = np.linspace(0.625,5.625,6)
            bar_x_250 = np.linspace(0.875,5.875,6)
            plt.bar(bar_x,activities_count,color='black',label='Total',width=0.9)
            plt.bar(bar_x_20,activities_count_20,label='20 Frame',width=0.22)
            plt.bar(bar_x_40,activities_count_40,label='40 Frame',width=0.22)
            plt.bar(bar_x_100,activities_count_100,label='100 Frame',width=0.22)
            plt.bar(bar_x_250,activities_count_250,label='250 Frame',width=0.22)
            plt.legend()
            plt.title(str(len(act_sub_li))+'-Nearest Neighbors (Unweighted)')
            plt.xlabel('Activity Type (abbrev.)')
            plt.ylabel('Counts (arb)')
            plt.xticks(bar_x,['box','clap','wave','jog','run','walk'])
            plt.savefig('test_img.png')
            if show and i%50 == 0:
                plt.show()
            else:
                print i
                plt.clf()
                plt.cla()
                plt.close()
            writer.append_data(imageio.imread('test_img.png'))
            os.remove('test_img.png')
    #        print activities_count
    if write_video:
        writer.close()
    return final_set_state


def plot_confusion_matrix(matrix,filename,data_fraction=1.0):
    '''
    Uses matplotlib to plot the heatmap of the confusion matrix.
    Args:
        matrix = the confusion matrix in question
        filename = file name (.png) to call the output file
        data_fraction = float fraction of the data (between 0 and 1) used to train
    Returns:
        None
    '''
    act_list = ['box','clap','wave','jog','run','walk']
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xticks(np.arange(len(act_list)))
    ax.set_yticks(np.arange(len(act_list)))
    ax.imshow(matrix,cmap='Blues')
    #plt.imshow(confusion_matrix_100,cmap='inferno')
    #ax.colorbar()
    ax.set_xticklabels(act_list,fontsize=14)
    ax.set_yticklabels(act_list,fontsize=14)
    for i in range(len(act_list)):
        for j in range(len(act_list)):
            ax.text(j, i, np.array(matrix)[i, j],
                ha="center", va="center", color="red",fontsize=20)
    plt.title('Confusion Matrix, '+str(int(100.0*data_fraction))+'% Training Set',fontsize=24)
    plt.savefig(filename)
    plt.show()


def read_training_mhi(fraction):
    '''
    Reads in the full training datasets, then takes the desired fraction to train with.
    Args:
        fraction = float between 0 and 1 representing the quantity of data to be used to train.
    Returns:
        Data and labels left after the reading in and exclusion of (1-fraction)*100% of the data
    '''
    ### READ IN COMPUTED DATA, 20 Frames ###
    with open('hu_vals_20_mhi.json') as json_data:
        hu_vals_20 = json.load(json_data)
        json_data.close()
    ### Filter out NaNs ###
    overall_hu_20 = []
    hu_20_labels = []
    for i in hu_vals_20.keys():
        hu_vals_df = pd.DataFrame(hu_vals_20[i])
        hu_vals_df.dropna
        hu_vals = np.array(hu_vals_df.values.tolist())
        train_num = int(fraction*hu_vals.shape[0])
        hu_vals = hu_vals[np.random.choice(hu_vals.shape[0], train_num, replace=False)]
        for j in hu_vals:
            overall_hu_20.append(j)
            hu_20_labels.append(i)
    overall_hu_20 = np.array(overall_hu_20).astype('int')*float(10**-12)
    
    ### READ IN COMPUTED DATA, 40 Frames ###
    
    with open('hu_vals_40_mhi.json') as json_data:
        hu_vals_40 = json.load(json_data)
        json_data.close()
    ### Filter out NaNs ###
    overall_hu_40 = []
    hu_40_labels = []
    for i in hu_vals_40.keys():
        hu_vals_df = pd.DataFrame(hu_vals_40[i])
        hu_vals_df.dropna
        hu_vals = np.array(hu_vals_df.values.tolist())
        train_num = int(fraction*hu_vals.shape[0])
        hu_vals = hu_vals[np.random.choice(hu_vals.shape[0], train_num, replace=False)]
        for j in hu_vals:
            overall_hu_40.append(j)
            hu_40_labels.append(i)
    overall_hu_40 = np.array(overall_hu_40).astype('int')*float(10**-12)
    
    ### READ IN COMPUTED DATA, 100 Frames ###
    
    with open('hu_vals_100_mhi.json') as json_data:
        hu_vals_100 = json.load(json_data)
        json_data.close()
    ### Filter out NaNs ###
    overall_hu_100 = []
    hu_100_labels = []
    for i in hu_vals_100.keys():
        hu_vals_df = pd.DataFrame(hu_vals_100[i])
        hu_vals_df.dropna
        hu_vals = np.array(hu_vals_df.values.tolist())
        train_num = int(fraction*hu_vals.shape[0])
        hu_vals = hu_vals[np.random.choice(hu_vals.shape[0], train_num, replace=False)]
        for j in hu_vals:
            overall_hu_100.append(j)
            hu_100_labels.append(i)
    overall_hu_100 = np.array(overall_hu_100).astype('int')*float(10**-12)
    
    ### READ IN COMPUTED DATA, 250 Frames ###
    
    with open('hu_vals_250_mhi.json') as json_data:
        hu_vals_250 = json.load(json_data)
        json_data.close()
    ### Filter out NaNs ###
    overall_hu_250 = []
    hu_250_labels = []
    for i in hu_vals_250.keys():
        hu_vals_df = pd.DataFrame(hu_vals_250[i])
        hu_vals_df.dropna
        hu_vals = np.array(hu_vals_df.values.tolist())
        train_num = int(fraction*hu_vals.shape[0])
        hu_vals = hu_vals[np.random.choice(hu_vals.shape[0], train_num, replace=False)]
        for j in hu_vals:
            overall_hu_250.append(j)
            hu_250_labels.append(i)
    overall_hu_250 = np.array(overall_hu_250).astype('int')*float(10**-12)
    return overall_hu_20,overall_hu_40,overall_hu_100,overall_hu_250,\
            hu_20_labels,hu_40_labels,hu_100_labels,hu_250_labels



def run_main(filename,true_state,fraction=0.9,knn_neighbors=3,out_file='test.mp4'):
    '''
    Takes a file as input and runs the desired analysis.
    Args:
        filename = string, what is the video file you wish to analyze?
        fraction = portion of the training dataset to select
        knn_neighbors = number of neighbors to include with the knn modeling
        out_file = what the saved file is to be called
    Returns:
        State most likely after the video has been analyzed.
    '''
    hu_info = read_training_mhi(fraction)
    print 'Files successfully read in!'
    #Training the model...
        
    num_neighbors = knn_neighbors
    
    knn20_hu_abbrev = NearestNeighbors(n_neighbors=num_neighbors)
    knn20_hu_abbrev.fit(np.array([i[:2] for i in hu_info[0]]))
    print 'Trained on 20 Frames!'
    
    knn40_hu_abbrev = NearestNeighbors(n_neighbors=num_neighbors)
    knn40_hu_abbrev.fit(np.array([i[:2] for i in hu_info[1]]))
    print 'Trained on 40 Frames!'
    
    knn100_hu_abbrev = NearestNeighbors(n_neighbors=num_neighbors)
    knn100_hu_abbrev.fit(np.array([i[:2] for i in hu_info[2]]))
    print 'Trained on 100 Frames!'
    
    knn250_hu_abbrev = NearestNeighbors(n_neighbors=num_neighbors)
    knn250_hu_abbrev.fit(np.array([i[:2] for i in hu_info[3]]))
    print 'Trained on 250 Frames!'
    knn_data = [knn20_hu_abbrev,knn40_hu_abbrev,knn100_hu_abbrev,knn250_hu_abbrev]
    return write_video(hu_info,knn_data,filename,true_state,out_video=out_file,show=True,write_video=True)


'''
These are the final confusion matrices which were generated
by iterating over all the videos at each of the states of the training
information. There was no code written to generate these automatically,
a counter was just put in place while all the videos were processed.
For ease, the results of this calculation have been included below.


confusion_matrix_100 = [[100,0,0,0,0,0],\
                        [0,100,0,0,0,0],\
                        [0,0,100,0,0,0],\
                        [0,0,0,100,0,0],\
                        [0,0,0,0,100,0],\
                        [0,0,0,0,0,100]]

confusion_matrix_90 = [[99,1,0,0,0,0],\
                       [0,100,0,0,0,0],\
                       [0,0,100,0,0,0],\
                       [0,0,0,98,2,0],\
                       [0,0,0,0,97,3],\
                       [0,0,0,1,0,99]]

confusion_matrix_60 = [[96,4,0,0,0,0],\
                       [2,98,0,0,0,0],\
                       [0,1,99,0,0,0],\
                       [0,0,0,93,3,4],\
                       [0,0,0,0,92,8],\
                       [0,0,0,0,1,99]]

confusion_matrix_30 = [[82,17,1,0,0,0],\
                       [5,93,2,0,0,0],\
                       [1,6,93,0,0,0],\
                       [0,0,0,78,5,17],\
                       [0,0,0,2,79,19],\
                       [0,0,0,0,4,96]]

confusion_matrix_10 = [[77,20,3,0,0,0],\
                       [5,92,3,0,0,0],\
                       [2,14,84,0,0,0],\
                       [0,0,0,47,6,47],\
                       [0,0,0,3,34,63],\
                       [0,0,0,1,5,94]]


To run this, simply copy one of the matrices to the bottom of the file and then
run the function plot_confusion_matrix(confusion_matrix_XXX, filename, data_fraction)
to get the appropriate heatmap (these are shown in the final video.)
'''



### GRADERS, EDIT AS NECESSARY HERE. YOU WILL NEED TO 
### EXAMPLE RUN, TO BE EDITED BY THE GRADER AS THEY SEE FIT.

fraction = 0.8 #Deciding the quantity of training data to use in the computation
#Select an action for the true state.
true_state = 'boxing'
path = 'input_files/'+true_state

for f in os.listdir(path)[0:1]: #Select the index of the video(s) to analyze
    filename = os.path.join(path,f)
    print run_main(filename,true_state)
    
    
    
    
