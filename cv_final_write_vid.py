'''
This file is only written in order to generate the final video required for submission.
It requires the pre-computed videos in order to create it, or else these must be
recalculated accordingly. Alter this file at your own risk.
-M. Schwarting
'''


import imageio
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

out_video = 'final_video_submission.mp4'

writer = imageio.get_writer(out_video, fps=30)



blank_sheet = np.zeros([720,720,3],np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX



blank_sheet1 = blank_sheet.copy()	
	

#### FRAME 1 ####
cv2.putText(blank_sheet1, 'Activity Recognition', (60, 250), font, 1.8, (255, 0, 0), 4,)
cv2.putText(blank_sheet1, 'Using MHIs and Hu Values', (60, 300), font, 1.4, (255, 0, 0), 4,)



cv2.putText(blank_sheet1, 'An Original Submission By:', (60, 500), font, 1.4, (255, 0, 0), 4,)
cv2.putText(blank_sheet1, 'Marcus Schwarting', (60, 550), font, 1.4, (255, 0, 0), 4,)
cv2.putText(blank_sheet1, 'meschw04@gatech.edu', (60, 600), font, 1.4, (255, 0, 0), 4,)

for i in range(0,50):
    writer.append_data(blank_sheet1)

#### FRAME 2 ####

blank_sheet2 = blank_sheet.copy()
cv2.putText(blank_sheet2, 'First Round of Testing:', (60, 300), font, 1.6, (255, 0, 0), 4,)
cv2.putText(blank_sheet2, 'Training is 100% of Dataset', (60, 400), font, 1.2, (255, 0, 0), 4,)

for i in range(0,50):
    writer.append_data(blank_sheet2)


#### VIDEO 1 ####

file_ext = 'test_boxing_all.mp4'

vid = imageio.get_reader(file_ext,  'ffmpeg')
len_vid = vid.get_length()
img_li = []
for num in range(0,len_vid):
    image = vid.get_data(num)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    writer.append_data(image)


#### FRAME 3 ####

blank_sheet3 = blank_sheet.copy()
cv2.putText(blank_sheet3, 'So Far, So Good.', (60, 300), font, 1.6, (255, 0, 0), 4,)
cv2.putText(blank_sheet3, 'Accuracy: 100%', (60, 400), font, 1.2, (255, 0, 0), 4,)
cv2.putText(blank_sheet3, '(as expected)', (60, 450), font, 1.2, (255, 0, 0), 4,)
cv2.putText(blank_sheet3, 'Now we will decrease', (60, 550), font, 1.2, (255, 0, 0), 4,)
cv2.putText(blank_sheet3, 'the training set size.', (60, 600), font, 1.2, (255, 0, 0), 4,)


for i in range(0,80):
    writer.append_data(blank_sheet3)



#### FRAME 4 ####

blank_sheet4 = blank_sheet.copy()
cv2.putText(blank_sheet4, 'Second Round of Testing:', (60, 300), font, 1.6, (255, 0, 0), 4,)
cv2.putText(blank_sheet4, 'Training is 90% of Dataset', (60, 400), font, 1.2, (255, 0, 0), 4,)

for i in range(0,50):
    writer.append_data(blank_sheet4)





#### VIDEO 2 ####

file_ext = 'test_handwaving_90.mp4'

vid = imageio.get_reader(file_ext,  'ffmpeg')
len_vid = vid.get_length()
img_li = []
for num in range(0,len_vid):
    image = vid.get_data(num)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    writer.append_data(image)



#### VIDEO 3 ####

file_ext = 'test_running_90.mp4'

vid = imageio.get_reader(file_ext,  'ffmpeg')
len_vid = vid.get_length()
img_li = []
for num in range(0,len_vid):
    image = vid.get_data(num)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    writer.append_data(image)


#### VIDEO 4 ####

file_ext = 'test_handclapping_90.mp4'

vid = imageio.get_reader(file_ext,  'ffmpeg')
len_vid = vid.get_length()
img_li = []
for num in range(0,len_vid):
    image = vid.get_data(num)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    writer.append_data(image)



#### FRAME 5 ####

blank_sheet5 = blank_sheet.copy()
cv2.putText(blank_sheet5, 'Accuracy: 98.8%', (60, 400), font, 1.2, (255, 0, 0), 4,)
cv2.putText(blank_sheet5, '(Oh yeah! It works!)', (60, 450), font, 1.2, (255, 0, 0), 4,)


for i in range(0,50):
    writer.append_data(blank_sheet5)



#### FRAME 6 ####

blank_sheet6 = blank_sheet.copy()
cv2.putText(blank_sheet6, 'Third Round of Testing:', (60, 300), font, 1.6, (255, 0, 0), 4,)
cv2.putText(blank_sheet6, 'Training is 60% of Dataset', (60, 400), font, 1.2, (255, 0, 0), 4,)

for i in range(0,50):
    writer.append_data(blank_sheet6)



#### VIDEO 5 ####

file_ext = 'test_walking_60.mp4'

vid = imageio.get_reader(file_ext,  'ffmpeg')
len_vid = vid.get_length()
img_li = []
for num in range(0,len_vid):
    image = vid.get_data(num)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    writer.append_data(image)


#### VIDEO 6 ####

file_ext = 'test_jogging_60.mp4'

vid = imageio.get_reader(file_ext,  'ffmpeg')
len_vid = vid.get_length()
img_li = []
for num in range(0,len_vid):
    image = vid.get_data(num)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    writer.append_data(image)

#### VIDEO 7 ####

file_ext = 'test_boxing_60.mp4'

vid = imageio.get_reader(file_ext,  'ffmpeg')
len_vid = vid.get_length()
img_li = []
for num in range(0,len_vid):
    image = vid.get_data(num)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    writer.append_data(image)



#### FRAME 7 ####

blank_sheet7 = blank_sheet.copy()
cv2.putText(blank_sheet7, 'Accuracy: 96.2%', (60, 400), font, 1.2, (255, 0, 0), 4,)
cv2.putText(blank_sheet7, '(Still doing well)', (60, 450), font, 1.2, (255, 0, 0), 4,)


for i in range(0,50):
    writer.append_data(blank_sheet7)


#### FRAME 8 ####

blank_sheet8 = blank_sheet.copy()
cv2.putText(blank_sheet8, 'A trend starts to develop.', (60, 400), font, 1.2, (255, 0, 0), 4,)
cv2.putText(blank_sheet8, 'Clapping, boxing, and waving', (60, 450), font, 1.2, (255, 0, 0), 4,)
cv2.putText(blank_sheet8, 'are follow together. Also', (60, 500), font, 1.2, (255, 0, 0), 4,)
cv2.putText(blank_sheet8, 'running, jogging, and walking', (60, 550), font, 1.2, (255, 0, 0), 4,)
cv2.putText(blank_sheet8, 'are very similar (or mistaken).', (60, 600), font, 1.2, (255, 0, 0), 4,)
cv2.putText(blank_sheet8, 'Default goes to walking.', (60, 650), font, 1.2, (255, 0, 0), 4,)


for i in range(0,120):
    writer.append_data(blank_sheet8)




#### FRAME 9 ####

blank_sheet9 = blank_sheet.copy()
cv2.putText(blank_sheet9, 'Fourth Round of Testing:', (60, 300), font, 1.6, (255, 0, 0), 4,)
cv2.putText(blank_sheet9, 'Training is 30% of Dataset', (60, 400), font, 1.2, (255, 0, 0), 4,)

for i in range(0,50):
    writer.append_data(blank_sheet9)



#### VIDEO 8 ####

file_ext = 'test_handwaving_30.mp4'

vid = imageio.get_reader(file_ext,  'ffmpeg')
len_vid = vid.get_length()
img_li = []
for num in range(0,len_vid):
    image = vid.get_data(num)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    writer.append_data(image)


#### VIDEO 9 ####

file_ext = 'test_walking_30.mp4'

vid = imageio.get_reader(file_ext,  'ffmpeg')
len_vid = vid.get_length()
img_li = []
for num in range(0,len_vid):
    image = vid.get_data(num)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    writer.append_data(image)



#### FRAME 10 ####

blank_sheet10 = blank_sheet.copy()
cv2.putText(blank_sheet10, 'Accuracy: 86.9%', (60, 400), font, 1.2, (255, 0, 0), 4,)
cv2.putText(blank_sheet10, '(Still doing well)', (60, 450), font, 1.2, (255, 0, 0), 4,)


for i in range(0,50):
    writer.append_data(blank_sheet10)




#### FRAME 11 ####

blank_sheet11 = blank_sheet.copy()
cv2.putText(blank_sheet11, 'Final Round of Testing:', (60, 300), font, 1.6, (255, 0, 0), 4,)
cv2.putText(blank_sheet11, 'Training is 10% of Dataset', (60, 400), font, 1.2, (255, 0, 0), 4,)

for i in range(0,50):
    writer.append_data(blank_sheet11)





#### VIDEO 10 ####

file_ext = 'test_running_10.mp4'

vid = imageio.get_reader(file_ext,  'ffmpeg')
len_vid = vid.get_length()
img_li = []
for num in range(0,len_vid):
    image = vid.get_data(num)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    writer.append_data(image)

file_ext = 'test_jogging_10.mp4'

vid = imageio.get_reader(file_ext,  'ffmpeg')
len_vid = vid.get_length()
img_li = []
for num in range(0,len_vid):
    image = vid.get_data(num)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    writer.append_data(image)

file_ext = 'test_handwaving_10.mp4'

vid = imageio.get_reader(file_ext,  'ffmpeg')
len_vid = vid.get_length()
img_li = []
for num in range(0,len_vid):
    image = vid.get_data(num)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    writer.append_data(image)


file_ext = 'test_handclapping_10.mp4'

vid = imageio.get_reader(file_ext,  'ffmpeg')
len_vid = vid.get_length()
img_li = []
for num in range(0,len_vid):
    image = vid.get_data(num)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    writer.append_data(image)


#### FRAME 12 ####

blank_sheet12 = blank_sheet.copy()
cv2.putText(blank_sheet12, 'Accuracy: 71.4%', (60, 400), font, 1.2, (255, 0, 0), 4,)
cv2.putText(blank_sheet12, 'Way better', (60, 450), font, 1.2, (255, 0, 0), 4,)
cv2.putText(blank_sheet12, 'than expected!', (60, 500), font, 1.2, (255, 0, 0), 4,)


for i in range(0,50):
    writer.append_data(blank_sheet12)

#### FRAME 13 ####

blank_sheet13 = blank_sheet.copy()
cv2.putText(blank_sheet13, "Let's take a look at", (60, 400), font, 1.2, (255, 0, 0), 4,)
cv2.putText(blank_sheet13, 'the confusion matrices:', (60, 450), font, 1.2, (255, 0, 0), 4,)


for i in range(0,50):
    writer.append_data(blank_sheet13)



#### FRAME 14 ####

blank_sheet14 = blank_sheet.copy()
cv2.putText(blank_sheet14, "Let's take a look at", (60, 400), font, 1.2, (255, 0, 0), 4,)
cv2.putText(blank_sheet14, 'the confusion matrices:', (60, 450), font, 1.2, (255, 0, 0), 4,)


for i in range(0,50):
    writer.append_data(blank_sheet14)

#### CONFUSION 100 ####

read_confusion_100 = cv2.imread('confusion_matrix_100.png')
read_confusion_100 = cv2.cvtColor(read_confusion_100, cv2.COLOR_BGR2RGB)
for i in range(0,50):
    writer.append_data(np.array(read_confusion_100))


#### CONFUSION 90 ####

read_confusion_90 = cv2.imread('confusion_matrix_90.png')
read_confusion_90 = cv2.cvtColor(read_confusion_90, cv2.COLOR_BGR2RGB)

for i in range(0,50):
    writer.append_data(read_confusion_90)


#### CONFUSION 60 ####

read_confusion_60 = cv2.imread('confusion_matrix_60.png')
read_confusion_60 = cv2.cvtColor(read_confusion_60, cv2.COLOR_BGR2RGB)

for i in range(0,50):
    writer.append_data(read_confusion_60)


#### CONFUSION 30 ####

read_confusion_30 = cv2.imread('confusion_matrix_30.png')
read_confusion_30 = cv2.cvtColor(read_confusion_30, cv2.COLOR_BGR2RGB)

for i in range(0,50):
    writer.append_data(read_confusion_30)

#### CONFUSION 10 ####

read_confusion_10 = cv2.imread('confusion_matrix_10.png')
read_confusion_10 = cv2.cvtColor(read_confusion_10, cv2.COLOR_BGR2RGB)

for i in range(0,50):
    writer.append_data(read_confusion_10)


#### FRAME 15 ####

blank_sheet15 = blank_sheet.copy()
cv2.putText(blank_sheet15, "Thanks for Watching!", (60, 400), font, 1.2, (255, 0, 0), 4,)
cv2.putText(blank_sheet15, '-M. Schwarting', (60, 550), font, 1.2, (255, 0, 0), 4,)


for i in range(0,80):
    writer.append_data(blank_sheet15)


writer.close()


