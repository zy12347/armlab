"""!
Class to represent the camera.
"""

import cv2
import time
import numpy as np
from PyQt4.QtGui import QImage
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from apriltag_ros.msg import *
from cv_bridge import CvBridge, CvBridgeError


class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.array([])

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.eye(3)
        self.extrinsic_matrix = np.eye(4)
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        #self.grid_x_points = np.arange(155, 1145, 50)
        #self.grid_y_points = np.arange(0, 660, 55)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275]]
        self.calibrate_h = np.identity(4)
        self.calibrate_flag = 0
        self.distort = np.array([])
        """ homo"""
        self.homography_matrix = np.identity(3)
        self.homography_matrix_2 = np.array([])
        """ block info """
        self.block_contours = np.array([])
        self.big_blocks = np.array([])
        self.small_blocks = np.array([])
        self.block_detections = np.array([])#pos and depth
        self.big_colors = np.array([])
        self.small_colors = np.array([])
        #self.big_block_angles = np.array([])#angles
        #self.small_block_angles = np.array([])#angles
        self.detectedflag = False
        self.depth_data =np.array([])
       # """color"""
        # self.color = list(({'id':'red','color':(10,10,127)},
        #                    {'id':'orange','color':(30,75,150)},
        #                    {'id':'yellow','color':(30,150,200)},
        #                    {'id':'green','color':(20,60,20)},
        #                    {'id':'blue','color':(100,50,0)},
        #                    {'id':'violet','color':(100,40,80)}))

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            # if self.calibrate_flag:
            #      frame = cv2.warpPerspective(frame,self.homography_matrix,(1280,720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        #print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def colors(self,hsv):
        colorbar = ['g','y','r','r','o','b','p']
        color_bounds_lower = np.array([[35,43,46],[20,43,46],[0,43,46],[156,43,46],[3,43,46],[80,43,46],[111,43,46]])
        color_bounds_upper = np.array([[77,255,255],[34,255,255],[2,255,255],[180,255,255],[19,255,255],[110,255,255],[155,255,255]])
        index = 0
        low = hsv>=color_bounds_lower
        low_ = np.sum(low,axis=1)
        up = hsv <= color_bounds_upper
        up_ = np.sum(up, axis=1)
        whole = low_+up_
        index = np.argmax(whole)

        return colorbar[index]

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        self.detectBlocksInDepthImage()
        # location = ((self.depth_data>0)*255).astype(np.uint8)

        image = self.VideoFrame[10:-50,80:-160]
        # print(type(self.VideoFrame))
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # ret, binary = cv2.threshold(location,127,255,0)
        # print(binary.shape)
        # kernel = np.ones((5, 5), 'uint8')
        # binary = cv2.dilate(binary, kernel)

        # _, contours,hierarchy= cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # self.block_contours = contours
        contours = self.block_contours
        big_blocks = [] #x,y,angle,color
        small_blocks = []
        centers = []
        big_angles = []
        small_angles = []
        big_colors = []
        small_colors = []
        colorbar ={'r':(255,0,0),'g':(0,255,0),'b':(0,0,255),'p':(255,0,255),'y':(255,255,0),'o':(255,128,0)}
        for pic,contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>600 and area<4000):
                x,y,w,h = cv2.boundingRect(contour)
                pos,size,angle = cv2.minAreaRect(contour)
                center = [int(x+w/2),int(y+h/2),self.depth_data[y][x]]
                hsv_value = hsv[center[1],center[0],:]
                # hsv_value = np.zeros(3)
                # print("hsv",hsv[center[1],center[0],:])
                # for i in range(30):
                #     for j in range(30):
                #         hsv_value = hsv_value + hsv[center[1]+i-15,center[0]+j-15,:]
                # hsv_value = hsv_value/900
                # co = self.colors(hsv[center[1],center[0],:])
                co = self.colors(hsv_value)
                block = [int(x+80+w/2),int(y+h/2+10),angle]
                if(area>1500 and area<4000):
                    big_blocks.append(np.array(block))
                    big_colors.append(co)
                    #angles.append(angle)
                    #centers.append(center)
                if(area>600 and area<1500):
                    small_blocks.append(np.array(block))
                    small_colors.append(co)
                    #angles.append(angle)
                    #centers.append(center)
                cv2.putText(self.VideoFrame,co+str(round(angle,2)),(x+80,y+10),cv2.FONT_HERSHEY_SIMPLEX,1.0,colorbar[co])
                #pos = (int(x+120),int(y+10))
                #endpoint = (int(pos[0]+40*np.cos(angle)),int(pos[1]+40*np.sin(angle)))
                #pos = (int(pos[0]),int(pos[1]))
                #endpoint = (int(pos[0]+40*np.cos(angle)),int(pos[1]+40*np.sin(angle)))
                #image = cv2.line(image,pos,endpoint,(0,255,0),9)
                #image = cv2.rectangle(image, (int(pos[0]), int(pos[1])), (int(pos[0]) + int(size[0]), int(pos[1] + size[1])), (0, 0, 255), 2)
                image = cv2.rectangle(self.VideoFrame,(x+80,y+10),(x+80+w,y+10+h),(0,0,255),2)
        self.big_blocks = np.array(big_blocks)
        self.small_blocks = np.array(small_blocks)
        self.big_colors = np.array(big_colors)
        self.small_colors = np.array(small_colors)
        #self.block_detections = np.array(centers)
        #self.block_angles = np.array(angles)
        #print(angles)
        #print(centers)
        #cv2.imwrite("detected.png",image)
        

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        lower=700
        upper=960
        depth_data = self.DepthFrameRaw[10:-50,80:-160].copy()
        self.depth_data = depth_data
        #print("depth1",depth_data)
        #print("depth_data",depth_data)
        # theta = np.arctan((model0.z-model3.z)/(((model0.y-model3.y)**2+(model0.x-model3.x)**2)**0.5))
        # theta = np.abs(theta)
        # print(self.intrinsic_matrix)
        # origin_x,origin_y = int(self.intrinsic_matrix[0][2]),int(self.intrinsic_matrix[1][2])
        # origin_z = depth_data[origin_x][origin_y]
        # origin_y -= origin_z*np.sin(theta)
        #print("shape0:",depth_data.shape[0])
        #print(type(depth_data[0][0]))
        #print("depth_shape",depth_data.shape)
        for x in range(depth_data.shape[0]):
                depth_data[x] = depth_data[x]-((x-300)/5)
        #print("depth2",depth_data)
        #print(depth_data[:,200])
        #print(depth_data)
        #rgb_data = self.VideoFrame[10:-50,120:-160]
        mask = np.zeros_like(depth_data,dtype = np.uint8)
        cv2.rectangle(mask,(1,4),(1278,720),0,cv2.FILLED)
        cv2.rectangle(mask,(150,13),(1146,658),255,cv2.FILLED)
        #cv2.rectangle(rgb_data,(1,4),(1278,720),(255,0,0),2)
        #cv2.rectangle(rgb_data,(150,13),(1146,658),(255,0,0),2)
        thresh = cv2.bitwise_and(cv2.inRange(depth_data,lower,upper),mask)
        #print("thresh",thresh)
        kernel = np.ones((2, 2), 'uint8')
        #thresh = cv2.dilate(thresh, kernel)
        _,self.block_contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        #cv2.drawContours(rgb_data,contours,-1,(0,255,255),3)
        #print("size",np.shape(self.block_contours))
        #self.blockcontourflag = True
        #self.depth_data = thresh

        cv2.imwrite("tgresgd.png",thresh)
        #cv2.imwrite("dawsd.png",rgb_data)
        ###ckp3.1
        pass

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame and
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
                
        """
       
        self.GridFrame = np.copy(self.VideoFrame)
        color = (255,0,0)
        thickness = 2
        extrinic = np.linalg.inv(self.calibrate_h)

        for i in range(len(self.grid_x_points)):
            for j in range(len(self.grid_y_points)):
                grid_x_y = extrinic.dot(np.array([self.grid_x_points[i],self.grid_y_points[j],1,1]).T)
                grid_x_y = (grid_x_y/grid_x_y[3])[0:3]
                grid_x_y = self.intrinsic_matrix.dot(grid_x_y)
                grid_x_y = grid_x_y/grid_x_y[2]
                grid_x_y = self.homography_matrix.dot(grid_x_y)
                grid_x_y = grid_x_y/grid_x_y[2]
                cv2.circle(self.GridFrame,(int(grid_x_y[0]),int(grid_x_y[1])),3,color,thickness)
        pass

class ImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            if self.camera.calibrate_flag:
                cv_image = cv2.warpPerspective(cv_image,self.camera.homography_matrix,(1280,720))
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image
        if self.camera.calibrate_flag and self.camera.detectedflag:
            self.camera.blockDetector()


class TagImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.TagImageFrame = cv_image


class TagDetectionListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, AprilTagDetectionArray,
                                        self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.tag_detections = data
        #for detection in data.detections:
        #print(detection.id[0])
        #print(detection.pose.pose.pose.position)


class CameraInfoListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, CameraInfo, self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.K, (3, 3))
        self.camera.distort = data.D
        #print(self.camera.intrinsic_matrix)


class DepthListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            if self.camera.calibrate_flag:
                cv_depth = cv2.warpPerspective(cv_depth,self.camera.homography_matrix,(1280,720))
            #cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        #self.camera.DepthFrameRaw = self.camera.DepthFrameRaw/2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_image_topic = "/tag_detections_image"
        tag_detection_topic = "/tag_detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        tag_image_listener = TagImageListener(tag_image_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        while True:
            rgb_frame = self.camera.convertQtVideoFrame()
            depth_frame = self.camera.convertQtDepthFrame()
            tag_frame = self.camera.convertQtTagImageFrame()
            self.camera.projectGridInRGBImage()
            grid_frame = self.camera.convertQtGridFrame()
            if ((rgb_frame != None) & (depth_frame != None)):
                self.updateFrame.emit(
                    rgb_frame, depth_frame, tag_frame, grid_frame)
            time.sleep(0.03)
            if __name__ == '__main__':
                cv2.imshow(
                    "Image window",
                    cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                cv2.imshow(
                    "Tag window",
                    cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Grid window",
                    cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(3)
                time.sleep(0.03)


if __name__ == '__main__':
    camera = Camera()
    videoThread = VideoThread(camera)
    videoThread.start()
    rospy.init_node('realsense_viewer', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
