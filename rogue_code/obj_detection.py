########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

# Defined Variables #
SEG_IMG = 'segmented_image.jpg' # used for saving the post-processed segmented image 
DEVICE = "cuda" # used to indicate that we want to generate the model using the gpu
MODEL_WEIGHT = './weights/FastSAM-s.pt' # the weights for fastSAM

# import Stereo Camera libraries #
import pyzed.sl as sl
import sys

# Import Image processing and Time libraries #
import cv2
import time
import signal
from matplotlib import pyplot as plt
import pickle
from PIL import Image
import torchvision.transforms as transforms
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import os 

# Import the Segmentation Model Libraries #
from FastSAM.fastsam import FastSAM, FastSAMPrompt

# Global Variables #
terminationFlag = 0 # used to terminate program
runningState = 0 # used to tell if svo or zed camera mode
segmentType = 0 # 0 = WODIS, 1 = FastSAM

CAMERA = 0
SVO = 1
IMG = 2

# This class contains all data regarding the current frame, such as the objects detected 
# their positions and a flag that inidicates whether we are safe .. this is is used so we can process
# it in the main control loop.
class FrameData:
    def __init__(self, objectPositions, image, segmentData, originalImage, boundingBoxes, locations):
        self.objectPositions = objectPositions
        self.image = image
        self.segmentData = segmentData
        self.originalImage = originalImage
        self.boundingBoxes = boundingBoxes
        self.locations = locations
    
    def get_object_positions(self):
        return self.objectPositions # return all objects detected in the current frame 

    def get_image(self):
        return self.image

    def get_seg_data(self):
        return self.segmentData

    def get_original_image(self):
        return self.originalImage
    
    def get_bounding_boxes(self):
        return self.boundingBoxes
    
    def get_locations(self):
        return self.locations

# Used to silence stdout !
class SilenceStdout:
    def __enter__(self):
        self.original_stderr = sys.stderr
        sys.stderr = open('/dev/null', 'w')  # On Unix-like systems, this discards stderr
        self.original_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')  # On Unix-like systems, this discards stdout

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.original_stdout
        sys.stderr.close()
        sys.stderr = self.original_stderr

# Camera class - this is used to detect objects with the zed camera and find the position of the objects relative to the boat #
class Camera:
    """
    Initialise all zed components and load the segmentation model
    """
    def __init__(self, source=CAMERA):
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.coordinate_units = sl.UNIT.CENTIMETER
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP 
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.realBoundingBoxes = []
        self.testWrite = open("/home/bluebouy/kalem_stuff/object_detection/localisation_testing/distances.txt", "a")
        self.frameCount = 0

        if source == SVO:
            self.init_params.set_from_svo_file("/home/bluebouy/kalem_stuff/object_detection/buoys/1687049277.svo")
        elif source == CAMERA:
            self.init_params.camera_resolution = sl.RESOLUTION.VGA # resolution is 1344 * 376

        # Open the camera #
        err = self.zed.open(self.init_params) 
        if err != sl.ERROR_CODE.SUCCESS:
            print("Failed: to open zed camera")
            exit(1)

        # Enable positional tracking with default parameters
        tracking_parameters = sl.PositionalTrackingParameters()
        err = self.zed.enable_positional_tracking(tracking_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        # initialise the image data frame for getting camera frames #
        self.image = sl.Mat()
        
        # Initialise the point cloud
        self.point_cloud = sl.Mat() # (width=512, height=384)

        # Initialise the runtime parameters #
        self.runtime_parameters = sl.RuntimeParameters() 

        # Generate the segmentation model #

        #seg_model = FastSAM(MODEL_WEIGHT) # -- This is how I generated the model
        #pickle.dump(seg_model, open('segmentation_model.pkl','wb'))
        with SilenceStdout():
            self.segmentation_model = pickle.load(open('segmentation_model.pkl','rb'))

        # Let's create our segmentation image plot !
        self.fig, self.ax = plt.subplots()
        self.fig.set_facecolor("#bababa")
        self.ax.set_title("Segmented Image")
        self.ax.set_xlabel("X pixels")
        self.ax.set_ylabel("Y pixels")
        self.text = self.ax.text(700, 500, "Detected Objects: " + str(0), fontsize=12, color='black', ha='right', va='top')

        plt.subplots_adjust(right=0.8)  # Adjust the right margin to make room for the legend
        self.textObjects = [] # used to store text objects
        self.boundingBoxes = [] # bounding boxes to display
        self.textObjects.append(self.text)

        # Lets Initialise Object tracking / detection #
        self.detection_parameters = sl.ObjectDetectionParameters()
        self.detection_parameters.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        self.detection_parameters.enable_tracking = False
        self.detection_parameters.enable_segmentation = False

        zed_error = self.zed.enable_object_detection(self.detection_parameters)
        if zed_error != sl.ERROR_CODE.SUCCESS:
            print("ERROR INITIALISING OBJECT DETECTION PARAMETERS")
            self.zed.close()
            exit(0)

        # Lets create our object parameters !
        self.objectData = sl.Objects()
        self.obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

        self.sensors_data = sl.SensorsData()


    """
    Close the camera and free the resources 
    """
    def close_camera(self):
        image.free(memory_type = sl.MEM.CPU)
        point_cloud.free(memory_type == sl.MEM.CPU)

        # Disable modules and close camera
        self.zed.disable_object_detection()
        self.zed.disable_positional_tracking()
        self.zed.close()

        print("Camera has closed :(")

    
    """
    grab the image and process it !
    """
    def grab_image(self):
        if (self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS):
            self.frameCount += 1
            # Retrieve frame from zed camera and convert it to our desired format#
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            image_data = self.image.get_data() 
            image_data = cv2.resize(image_data, (512, 384)) # resize to reduce computation costs
            image_data = cv2.cvtColor(image_data, cv2.COLOR_RGBA2RGB) # convert image to RGB format since thats what our image model uses

            # Retrieve the point cloud so we can determine location and size of object #
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)

            # Generate the segmentation from the image #
            with SilenceStdout():
                seg_result = self.segmentation_model(image_data, device=DEVICE, retina_masks=True, imgsz=(512, 384), conf=0.4, iou=0.7)
                prompt_process = FastSAMPrompt(image_data, seg_result, device=DEVICE)
                result = prompt_process.everything_prompt() 

                # Show the segmented image #
                prompt_process.plot(annotations=result, output_path=SEG_IMG)
            processed_img = cv2.imread(SEG_IMG)

            self.zed.get_sensors_data(self.sensors_data, sl.TIME_REFERENCE.IMAGE)
            zed_imu = self.sensors_data.get_imu_data() # get the imu data to calculate the position 
            
            # Get IMU orientation
            zed_imu_pose = sl.Transform()
            ox = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[0], 3) 
            oy = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[1], 3)
            oz = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[2], 3)
            ow = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[3], 3)

            print("IMU Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))
            # Get IMU acceleration
            acceleration = [0,0,0]
            zed_imu.get_linear_acceleration(acceleration)
            ax = round(acceleration[0], 3)
            ay = round(acceleration[1], 3)
            az = round(acceleration[2], 3)

            if len(self.textObjects) > 0:
                for obj in self.textObjects:
                    obj.remove() # remove the text from  the plot !
                self.textObjects.clear() # now clear the list

            # We found some objects ! lets process them
            bounding_boxes = []
            if prompt_process.results is not None: 
                bounding_boxes = self.extract_bounding_boxes(prompt_process.results[0].masks.xy)
                #for box in bounding_boxes:
                #    bbox = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
                #    self.ax.add_patch(bbox)
                #    self.boundingBoxes.append(bbox)

                text = self.ax.text(700, 500, "Detected Objects: " + str(len(prompt_process.results[0])), fontsize=12, color='black', ha='right', va='top')
                self.textObjects.append(text)

                # now we want o ingest the positions
                objects_in = []
                order = 1
                name = "/home/bluebouy/kalem_stuff/object_detection/localisation_testing/" + "frame_" + str(self.frameCount) + "/" # the name of the path to be used 

                if not os.path.exists(name):
                    os.makedirs(name) # make the directory if it does not exist

                # Set objects up with bounding box locations of objects
                for i in bounding_boxes:
                    self.remove_bounding_boxes() # only draw one at a time
                    self.draw_bounding_box([i]) # draw one at a time
                    plt.savefig(name + str(order))
                    tmp = sl.CustomBoxObjectData()
                    tmp.is_grounded = False # since it doesnt just move on 2D plane
                    tmp.probability = 0.8 # lets just go with a high confidence
                    tmp.unique_object_id = str(order) # gene rate a unique id for the object
                    tmp.label = order # lets just call it 0
                    A, B, C, D = [i[0], i[1]], [i[2], i[1]], [i[0], i[3]], [i[2], i[3]]
                    tmp.bounding_box_2d = np.array([A, B, C, D])
                    objects_in.append(tmp)
                    order += 1
                
                # now lets ingest it 
                self.zed.ingest_custom_box_objects(objects_in)
                self.zed.retrieve_objects(self.objectData, self.obj_runtime_param)
                count = 0

                objectPositions = []
                labels = []
                for obj in self.objectData.object_list:
                    labels.append(obj.raw_label)
                    objectPositions.append(obj.bounding_box)
                    print("Object:", count, "Position:", obj.bounding_box)
                    count += 1

                processed_file_name = "/home/bluebouy/kalem_stuff/object_detection/localisation_testing/segmented_images/" + str(self.frameCount) + ".jpg"
                orig_file_name = "/home/bluebouy/kalem_stuff/object_detection/localisation_testing/original_images/" + str(self.frameCount) + ".jpg"

                cv2.imwrite(orig_file_name, image_data) # original file to add to saved folders 
                cv2.imwrite(processed_file_name, processed_img) # write the processed image
                self.testWrite.write("\n------------" + str(self.frameCount) + "------------\n")
                self.testWrite.write("labels:" + str(labels)) # add the labels 
                
                self.testWrite.write(str(objectPositions)) # now lets write the object positions 

                # Now show the annotated frame ! #
                self.ax.imshow(processed_img)
                plt.draw()
                plt.pause(0.05)

                locations = self.find_object_positions(prompt_process.results[0].masks.xy)

                return FrameData(objectPositions, processed_img, prompt_process.results[0].masks.xy, image_data, bounding_boxes, locations) # lets return the image 

    # Draw a bounding box around the object we are going to crash with 
    def draw_bounding_box(self, bboxes: list):
        self.remove_bounding_boxes() # remove the bounding boxes from the list s
        for bbox in bboxes:
            drawBbox = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor='r', facecolor='none')
            self.ax.add_patch(drawBbox)
            self.boundingBoxes.append(drawBbox)

    # Remove all bounding boxes from the figure
    def remove_bounding_boxes(self):
        if len(self.boundingBoxes) > 0:
            for bbox in self.boundingBoxes:
                bbox.remove()
            self.boundingBoxes.clear() # clear the list 

    def extract_object_positions(self, masks: list, point_cloud: list):
        distances = []

        for mask in masks: # iterate through every mask 
            return mask

        return distances

    """
    extract the bounding boxes for all detected objects and return an array countaing the bounding boxes
    """
    def extract_bounding_boxes(self, masks: list) -> list:
        bounding_boxes = []
        for mask in masks:
            maskArray = np.array(mask) # convert to a np array so we can work with it 

            x_values = [point[0] for point in maskArray]
            y_values = [point[1] for point in maskArray]

            x_min = int(np.min(x_values))
            x_max = int(np.max(x_values))
            y_min = int(np.min(y_values))
            y_max = int(np.max(y_values))

            bounding_boxes.append((x_min, y_min, x_max, y_max)) # output in x1, y1, x2, y2 format 
        
        return bounding_boxes


    # masks format
    def find_object_positions(self, masks: list) -> list:
        counter = -1
        maskCount = 0

        xyzPositions = []

        PIXEL_RESOLUTION = 1
        testCounter = 0
        self.realBoundingBoxes = [] # reset it 

        # lets iterate through every single mask
        for mask in masks:
            
            maskCount += 1
            count = -1
            objectPositions = [] # reset the  list 

            for x, y in mask: # iterate through the x,y positions of the mask
                count += 1
                if count % PIXEL_RESOLUTION == 0: # every PIXEL_RESOLUTION pixels 
                    point3D = self.point_cloud.get_value(x, y)[1] # lets get the 3d position of every single pixel in the mask 
                    print(point3D)
                    point3D = point3D[:-1] # remove the RGB element
                    purgeValues = False
                    for value in point3D:
                        if np.isnan(value) or np.isinf(value) or value == 0.0: # check for nan and -inf/+inf
                            purgeValues = True
                            break # we do not want t count this value due to nan or inf properties
                    if purgeValues:
                        continue
                    # good value, lets add it to our array

                    objectPositions.append(point3D) # remove the RGB element, we only want x, y, z
                    testCounter += 1
            min_values = []
            if len(objectPositions) > 0:
                objectPositions = np.array(objectPositions)
                print("o: ", objectPositions)
                abs_x_vals = np.abs(objectPositions[:, 0]) # get absolute value of x values
                abs_z_vals = np.abs(objectPositions[:, 2]) # get absolute value of z values

                # we have finished calculating 1 mask
                min_z = abs_z_vals.argmin()
                min_x = min_z = abs_x_vals.argmin()
                min_values = [objectPositions[min_z], objectPositions[min_x]]
                
                z_vals = objectPositions[:, 2]
                x_vals = objectPositions[:, 0]
                y_vals = objectPositions[:, 1]

                max_z = z_vals.argmax()
                min_z = z_vals.argmin()
                max_x = x_vals.argmax()
                min_x = x_vals.argmin()
                max_y = y_vals.argmax()
                min_y = y_vals.argmin()

                self.realBoundingBoxes.append([min_x, max_x, min_y, max_y, min_z, max_z])
            xyzPositions.append(min_values) # append the lowest two
        
        return xyzPositions