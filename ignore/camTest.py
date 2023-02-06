import cv2
import tensorflow as tf
import numpy as np

vid = cv2.VideoCapture(0)
while(True):
    ret, frame = vid.read()
    model5 = tf.keras.models.load_model('finalModel\model5.h5')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('video', gray)
    img = cv2.resize(gray, (28, 28))
    img = img.reshape((28, 28, 1))
    img = img / 255
    img = np.array([img])
    pred = model5.predict(img)
    pred = np.argmax(pred)
    print(pred)
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()