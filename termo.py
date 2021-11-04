# %%
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

# %%
mean_array = np.zeros((3, frame_count))
i = 0
# Abre el video y lo muestra
while(cap.isOpened()):
  ret, frame = cap.read()
  # frame es un numpy array de (1000,1000,3)
  #frame2 = frame[:,:,1]  # extrae el primer color
  #mean = np.mean(frame2)
  mean_array[0, i] = np.mean(frame[:,:,0])
  mean_array[1, i] = np.mean(frame[:,:,1])
  mean_array[2, i] = np.mean(frame[:,:,2])
  
  #if i<10:
  #    print (type(frame), str(frame.shape))
  #else:
  #    break
  #print(i)
  i+=1    
  #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # filtro
  cv2.imshow('frame',frame)

  
  if cv2.waitKey(1) & 0xFF == ord('q'):
      # Se puede dejar de correr escribiendo Q
    break

cap.release()
cv2.destroyAllWindows()
# %%
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 1
plt.rcParams['grid.color'] = "#cccccc"
plt.rcParams.update({"font.size": 10, "font.family": "serif"})

frame_array = np.arange(0, frame_count, 1)
fig, ax = plt.subplots(figsize=(5,3.5))
ax.plot(frame_array, mean_array[0], color='blue', label='blue')
ax.plot(frame_array, mean_array[1], color='green', label='green')
ax.plot(frame_array, mean_array[2], color='red', label='red')
ax.legend()
ax.set_xlabel('Frame')
ax.set_ylabel('Mean intensity')
fig.show()
# %%


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