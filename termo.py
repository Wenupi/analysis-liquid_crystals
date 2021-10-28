import numpy as np
import cv2
import matplotlib.pyplot as plt

path = '/Users/javier/Documents/university/6th-semester/'+\
       'experimental-physic2/optic-lab/video-analysis/videos/'

cap = cv2.VideoCapture(path+'image_5.avi')  # selección del video
fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # cantidad de frames
duration = frame_count/fps

# Print de la información
print('fps = ' + str(fps))
print('number of frames = ' + str(frame_count))
print('duration (S) = ' + str(duration))
minutes = int(duration/60)
seconds = duration%60
print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

mean_array = np.zeros((1,frame_count))
i = 0
# Abre el video y lo muestra
while(cap.isOpened()):
  ret, frame = cap.read()
  # frame es un numpy array de (1000,1000,3)
  frame2 = frame[:,:,1]  # extrae el primer color
  mean = np.mean(frame2)
  mean_array[0, i] = mean
  print(mean)
  #if i<10:
  #    print (type(frame), str(frame.shape))
  #else:
  #    break
  #print(i)
  i+=1    
  #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # filtro
  #cv2.imshow('frame',frame)

  
  if cv2.waitKey(1) & 0xFF == ord('q'):
      # Se puede dejar de correr escribiendo Q
    break

frame_array = np.arange(0, frame_count, 1)
fig, ax = plt.subplots(figsize=(3.5,3.5))
ax.plot(frame_array, mean)
fig.show()

#cap.release()
#cv2.destroyAllWindows()

#cap = cv2.VideoCapture(0)
#
#while True:
#    ret, frame = cap.read()
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#    
#    cv2.imshow('frame',frame)
#    cv2.imshow('gray',gray)
#    cv2.imshow('hsv',hsv)
#
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
#cap.realease()
#cv2.destroyAllWindows()