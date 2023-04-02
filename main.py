import os
import numpy as np
import json
import cv2

def working_paths(library, dnn):
    """
    Function to define the working paths according to the AI library and NN model we want to use.
    Inputs:
        * library: (str) Defines the AI library we want to use. Depending on the library (Caffe,
        Tensorflow, Darknet, etc), the model dependencies will change.
        * dnn: (str) Model we want to use.
    Outputs:
        * (tuple) Model working paths.
        * (list) Images testing paths.
    """
    # Get current directory
    current_dir = os.getcwd()
    # Test images path
    img_list = ['test1.jpg','test2.jpg','test3.jpg', 'test4.jpg']
    test_imgs_path = [os.path.join(current_dir, 'myFiles', 'imgs', file_name) for file_name in img_list]
    # Models path
    models_path = os.path.join(current_dir, 'myFiles', 'models')
    # Define the IA library
    if library == 'CAFFE':
        caffe_path = os.path.join(models_path, 'caffe')
        if dnn == 'AlexNet':
            architecture_path = os.path.join(caffe_path, 'bvlc_alexnet.prototxt')
            weights_path = os.path.join(caffe_path, 'bvlc_alexnet.caffemodel')
            classes_path = os.path.join(models_path, 'imagenet_classes.json')
            scalefactor=1; size=(224, 224); mean=(104, 117, 123); swapRB=False
            blob = (scalefactor, size, mean, swapRB)
        elif dnn == 'GoogleNet':
            architecture_path = os.path.join(caffe_path, 'bvlc_googlenet.prototxt')
            weights_path = os.path.join(caffe_path, 'bvlc_googlenet.caffemodel')
            classes_path = os.path.join(models_path, 'imagenet_classes.json')
            scalefactor=1; size=(224, 224); mean=(104, 117, 123); swapRB=False
            blob = (scalefactor, size, mean, swapRB)
        elif dnn == 'VGG16':
            architecture_path = os.path.join(caffe_path, 'VGG_ILSVRC_16_layers.prototxt')
            weights_path = os.path.join(caffe_path, 'VGG_ILSVRC_16_layers.caffemodel')
            classes_path = os.path.join(models_path, 'imagenet_classes.json')
            scalefactor=1; size=(224, 224); mean=(104, 117, 123); swapRB=False
            blob = (scalefactor, size, mean, swapRB)
        elif dnn == 'VGG19':
            architecture_path = os.path.join(caffe_path, 'VGG_ILSVRC_19_layers.prototxt')
            weights_path = os.path.join(caffe_path, 'VGG_ILSVRC_19_layers.caffemodel')
            classes_path = os.path.join(models_path, 'imagenet_classes.json')
            scalefactor=1; size=(224, 224); mean=(104, 117, 123); swapRB=False
            blob = (scalefactor, size, mean, swapRB)

        return (architecture_path, weights_path, classes_path, blob), test_imgs_path
    
    elif library == 'TF':
        tf_path = os.path.join(models_path, 'tensorflow')

def net(img_path, model, n):
    """
    Function that allows us to classify an image using only OpenCV.DNN().
    Inputs:
        * img_path: (str) Test image path.
        * model: (tuple) Model working paths.
        * n: (int) Number of classes to display when classifying the image.
    """

    architecture, weights, classes, blob = model
    scalefactor, size, mean, swapRB = blob
    # ------- LOAD THE MODEL -------
    net = cv2.dnn.readNetFromCaffe(architecture, weights)
    classes = json.load(open(classes, 'r'))

    # ------- READ THE IMAGE AND PREPROCESSING -------
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, size)
    # Create a blob
    blob = cv2.dnn.blobFromImage(img_resized, scalefactor, size, mean, swapRB)
    
    # ------- DETECTIONS AND PREDICTIONS ----------
    net.setInput(blob)
    predictions = net.forward()
    print(predictions)
    # Get the n better results
    indexes = np.argsort(predictions[0])[::-1][:n]

    print('\nCLASSIFICATION')
    for i in range(n):
        label = classes[str(indexes[i])]
        conf = round(predictions[0][indexes[i]]*100, 2)
        print(f'{i}. {label}={conf}%')
 
    # ------- SHOW RESULTS -------
    cv2.imshow("Image", img)

    # ------- IF SOME KEY PRESSED CLOSE IMAGE -------
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # ------- WORK PATHS -------
    model, imgs = working_paths(library='CAFFE', dnn='squeezenet')    

    # -------  MAIN -------
    for img in imgs:
        net(img_path=img, model=model, n=2)

    print(' *** END PROGRAM ***\n')