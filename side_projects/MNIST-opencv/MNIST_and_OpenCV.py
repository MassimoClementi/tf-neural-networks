#!venv/bin/python3

'''
    Author: Massimo Clementi
    Date:   2021-04-08
    
    From the webcam feed, detect regions of interest and then run the
    self-trained NN to obtain the digit and the prediction confidence
    
'''



print('Importing modules...')
import numpy as np
import cv2
import tensorflow as tf



def new_model():
    # This must be the same model defined in:
    #   python_scripts/2_MNIST_dataset_and_CNNs.py
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            filters=20,
            kernel_size=5,
            activation='relu',
            data_format='channels_last',
            input_shape=(28,28,1),
        ),
        tf.keras.layers.MaxPool2D(
            pool_size=(2,2)
        ),
        tf.keras.layers.Conv2D(
            filters=50,
            kernel_size=5,
            activation='relu'
        ),
        tf.keras.layers.MaxPool2D(
            pool_size=(2,2)
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            units=500,
            activation='tanh'
        ),
        tf.keras.layers.Dense(
            units=10,
            activation='tanh'
        ),
        tf.keras.layers.Softmax()
    ])
    return model


    
def processing_results(model, data):
    
    try:
        
        # Patch image processing
               
        '''
        _, data = cv2.threshold(
            data,
            0,
            255,
            cv2.THRESH_BINARY+cv2.THRESH_OTSU
        )
        '''
        
        data = cv2.medianBlur(data, 3)
    
        data = cv2.resize(
            data,
            (28,28),
            interpolation=cv2.INTER_AREA
        )
    except:
        return 0, 0, data
    
    # For cv2 the dn of the digit are related to the color black (0),
    # however the NN is trained on the MNIST dataset where digits are
    # related to the color white (1). As a consequence it is required
    # to invert the dn of the patch before feeding it into the NN
    data = 1 - (data.astype(float) / np.max(data))
    
    dummy_batch = np.array(data).reshape([1,28,28,1])
    #print('data shape:',dummy_batch.shape)
    res = model(dummy_batch).numpy()
    num_hat = np.argmax(res[0])     # prediction of what number
    conf = res[0,num_hat]           # confidence of the prediction
    
    return num_hat, conf, data
    



def main():
    print('Accessing webcam feed...')
    cap = cv2.VideoCapture(1)
    
    # Create model and load weights
    print('Loading neural network...')
    model = new_model()
    model.load_weights('saved_models/2_MNIST_dataset_and_CNNs/CNN').expect_partial()
    
    print('All good!')
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        ''' Process image to detect the areas to investigate with the NN '''
        gray_proc = cv2.equalizeHist(gray)
        gray_proc = cv2.GaussianBlur(gray_proc, (5,5), cv2.BORDER_DEFAULT)
        gray_proc = cv2.Canny(gray_proc,10,200)
        gray_proc = cv2.morphologyEx(gray_proc, cv2.MORPH_CLOSE ,kernel=np.ones((30,30), np.uint8))
        contours, _ = cv2.findContours(gray_proc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
        boxes = []
        for cont in contours:
            # Filter out small contours
            if cv2.contourArea(cont) > 30 and cv2.contourArea(cont) < 1280*720/9:
                box = cv2.boundingRect(cont)
                
                # Filter out boxes that do not have square-ish shape
                w_h_ratio = box[2]/box[3]
                if w_h_ratio < 2 and w_h_ratio >0.5:
                    boxes.append(box)
                    '''
                    cv2.rectangle(
                        gray,
                        box,
                        [0,0,255]
                    )
                    '''
        
        
        for box in boxes:
        #for i, box in enumerate(boxes):
            x,y,w,h = box
            
            # Apply some offsets
            offs = 20
            try:
                x = x - offs
                y = y - offs
                w = w + 2*offs
                h = h + 2*offs
            except:
                pass
            
            '''
            cv2.rectangle(
                gray,
                (x, y),
                (x+w, y+h),
                [0,0,255]
            )
            '''
            
            num_hat, conf, proc_crop = processing_results(
                model=model,
                data=gray[y:y+h, x:x+w]   
            )
            
            if conf > 0.15:
                cv2.rectangle(
                    gray,
                    (x,y),
                    (x+w, y+h),
                    color=(0,0,255)
                )
                cv2.putText(
                    gray,
                    str(num_hat)+', conf '+'{:.2f}'.format(conf),
                    (x+15,y+5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color=(0,0,255)
                )
            
                win_name = 'focus'
                #win_name = 'focus'+str(i)
                cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
                cv2.resizeWindow(
                        win_name,
                        int(28*6),
                        int(28*6)
                )
                cv2.imshow(win_name,proc_crop)
        

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
main()