# TopologyAware
Code of "Topology-Aware Loss Function for Aorta and Great Vessel Segmentation in Computed Tomography Images" paper. Pre-print paper is uploaded here.

To run this code:
1. put the base and our folders into the same folder -- or you may change the addresses in our folder. You will go from our to base --
2. prepare your data:
     a. we have 256x256 ct scan dicom images (you can see the examples in the paper)
     b. gold images binary segmentations (check the paper)
     c. you need to create persistent homology diagram numbers for gold images and store them as a numpy file.
         For this: create dgms0 and dgms1 folders in your gold folder and run this code snippet with correct folder addresses :)
   
         *****
              import numpy as np
              import skimage.segmentation as sg
              import cv2
              import os
              import ripserplusplus as rpp
              
              dr='goldsBinaryAll'
              
              image_names = os.listdir(dr) 
              image_names.remove('dgms0')
              image_names.remove('dgms1')
              for name in image_names:
                  im = cv2.imread(dr+'/'+name,cv2.IMREAD_GRAYSCALE)
                  pairs = getPairs(im)
                  
                  imDgms0 = pairs[0]
                  imDgms0 = imDgms0.tolist()
                  imDgms0 = np.array(imDgms0)
                  
                  with open(dr+'/dgms0/'+name.split('.')[0]+'.npy', 'wb') as f:
                      np.save(f, imDgms0)
                      
                  imDgms1 = pairs[1]
                  imDgms1 = imDgms1.tolist()
                  imDgms1 = np.array(imDgms1)
                  
                  with open(dr+'/dgms1/'+name.split('.')[0]+'.npy', 'wb') as f:
                      np.save(f, imDgms1)          
         *****
     d. we use different homology group diagrams for different types of vessels; hence we need to know the kind in training time. We also use it in the network score calculations. For this, we used a JSON file named vesseltypes.json (e.g., {"aorta01_100.dcm":"s","aorta01_101.dcm":"s","aorta01_102.dcm":"s","aorta01_103.dcm":"s","aorta01_104.dcm":"s","aorta01_105.dcm":"s"} but much longer). You will need to replace it, or you use each diagram for every image and not use the file (you should change the code for this)
       
4. You should change the data addresses in the base/UnetRun.py and our/TopAwareUNetRun.py files. Also, check all python files and put the correct address for the JSON file (if you use it).
5. in the base folder, you should create 20_excels, 20_weights, 25_excels, 25_weights, 30_excels, 30_weights, excels, weights, graphs folders. Our uses 25_excels, 25_weights but you can change it. We start our training from the 25th epoch of the base model. In the end of the training, you will see the scores of the base model in excels folder and graphs in the graphs folder.
6. you should create excels, weights, and graphs folders in our folder. At the end of the training, you will see the scores in excels folder and graphs in the graphs folder.

Note that here run nums are 1 (num_of_runs = 1). Just to see in 1 run quickly. However, in our implementation, we run 5 times and get the mean and standard deviation.

If you have any questions, feel free to write sozcelik19@ku.edu.tr or seherozcelik@gmail.com.

Seher.
