"""!
The state machine that implements the logic.
"""
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
import rospy
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import Point
import cv2

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        ###Checkpoint1.4 used
        self.study = 0 #record time of repeating wayplan in ckp1_4, if repeated 5 times, ckp1_4 would be cleared
        self.ckp1_4_arm=list()#record the wayplan for checkpoint 1.4
        self.gripper_state=[]
        self.ckp1_4_gripper=list()
        self.mydepth = np.array([])
        #self.tagmsg = [Point(0,0,0),Point(0,0,0),Point(0,0,0),Point(0,0,0)]
        self.detect_grab_flag = False
        ###checkpoint 2.1 used
        self.waypoints = [
             [-np.pi/8,      0.0,     0.0,      0.0,     0.0],
             [np.pi/8,       0.0,     0.0,      0.0,     0.0],
             [0,      -np.pi / 8,     0.0,     0.0,      0.0],
             [0,       np.pi / 8,     0.0,     0.0,      0.0],
             [0.0,           0.0, -np.pi / 8,     0.0,     0.0, ],
             [0,          0.0,     np.pi / 8,     0.0,     0.0, ],
             [0,              0,      0,     -np.pi / 8,     0.0],
             [0,              0,      0,     np.pi / 8,     0.0],
             [0,         0,     0,      0.0,     -np.pi / 5],
             [0,         0,     0,      0.0,     np.pi / 5]]
        

        ###checkpoint 3.2 used
        # self.waypoints = [
        #     [-np.pi/2,       -0.5,      -0.3,            0.0,       0.0],
        #     [0.75*-np.pi/2,   0.5,      0.3,      0.0,       np.pi/2],
        #     [0.5*-np.pi/2,   -0.5,     -0.3,     np.pi / 2,     0.0],
        #     [0.25*-np.pi/2,   0.5,     0.3,     0.0,       np.pi/2],
        #     [0.0,             0.0,      0.0,         0.0,     0.0],
        #     [0.25*np.pi/2,   -0.5,      -0.3,      0.0,       np.pi/2],
        #     [0.5*np.pi/2,     0.5,     0.3,     np.pi / 2,     0.0],
        #     [0.75*np.pi/2,   -0.5,     -0.3,     0.0,       np.pi/2],
        #     [np.pi/2,         0.5,     0.3,      0.0,     0.0],
        #     [0.0,             0.0,     0.0,      0.0,     0.0]]
        
        

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()
        
        if self.next_state == "record":
            self.record()
        
        if self.next_state == "detect":
            self.detect()
        
        if self.next_state == "big_small":
            self.big_small()

        if self.next_state == "line":
            self.line()

        if self.next_state == "stack":
            self.stack()

        if self.next_state == "nstack":
            self.nstack()


    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.status_message = "State: Execute - Executing motion plan"
        with open("position.txt",'w') as f:
            f.write(str(self.ckp1_4_arm))
        
        with open("gripper.txt",'w') as f:
            f.write(str(self.ckp1_4_gripper))

        if self.study>0:
            for i in range(len(self.ckp1_4_arm)):
                self.rxarm.set_joint_positions(self.ckp1_4_arm[i],moving_time=2.0,accel_time=0.5,blocking=True)
                if (self.ckp1_4_gripper[i]):
                    self.rxarm.close_gripper()
                else:
                    self.rxarm.open_gripper()
                rospy.sleep(2)
            self.study +=1
            if (self.study == 5):
                self.ckp1_4_arm = []
                self.ckp1_4_gripper = []
                self.study = 0
        else:
            for position in self.waypoints:
                self.rxarm.set_joint_positions(position,moving_time=2.0,accel_time=0.5,blocking=True)
                rospy.sleep(2)
        self.next_state = "idle"

    def record(self):
        """!
        @brief      NEW function: for record the waypoint plan in checkpoint 1.4
        """
        self.status_message = "State: Record -Recording waypoint"
        self.study = 1
        self.ckp1_4_arm.append(self.rxarm.get_positions())
        # self.gripper_state.append(self.rxarm.get_gripper_position())
        print(self.rxarm.get_gripper_position())
        if (self.rxarm.get_gripper_position()<0.05):#gripper close
            self.ckp1_4_gripper.append(1)
        else:
            self.ckp1_4_gripper.append(0)
        print(self.ckp1_4_gripper)
        ###ckp3.1
        self.next_state = "idle"

    """
    callback
    """


    # def call_back_String(self,msg):
    #     print("received")
    #     self.tagmsg[0] = msg.detections[0].pose.pose.pose.position
    #     self.tagmsg[1] = msg.detections[1].pose.pose.pose.position
    #     self.tagmsg[2] = msg.detections[2].pose.pose.pose.position
    #     self.tagmsg[3] = msg.detections[3].pose.pose.pose.position

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""
        self.status_message = "Calibration - start Calibration"
        self.camera.calibrate_flag = 1

        
        for i in range(4):
            print(self.camera.tag_detections.detections[i].id[0])
            if self.camera.tag_detections.detections[i].id[0] == 3:
                model0 = self.camera.tag_detections.detections[i].pose.pose.pose.position
            if self.camera.tag_detections.detections[i].id[0] == 7:
                model1 = self.camera.tag_detections.detections[i].pose.pose.pose.position
            if self.camera.tag_detections.detections[i].id[0] == 4:    
                model2 = self.camera.tag_detections.detections[i].pose.pose.pose.position
            if self.camera.tag_detections.detections[i].id[0] == 8:    
                model3 = self.camera.tag_detections.detections[i].pose.pose.pose.position
            # if self.camera.tag_detections.detections[i].id[0] == 6:    
            #      model4 = self.camera.tag_detections.detections[i].pose.pose.pose.position
        camera_points = np.array([(model0.x,model0.y,model0.z),(model1.x,model1.y,model1.z),(model2.x,model2.y,model2.z),(model3.x,model3.y,model3.z)],dtype = np.float32)
        #camera_points = np.array([(model0.x,model0.y,model0.z),(model1.x,model1.y,model1.z),(model2.x,model2.y,model2.z),(model3.x,model3.y,model3.z),(model4.x,model4.y,model4.z)],dtype = np.float32)
        #print(camera_points)
        #print("camera_points",camera_points)
        #z_pos = np.array([model0.z,model1.z,model2.z,model3.z])
        z_pos = np.array([model0.z,model1.z,model2.z,model3.z])
        #print(self.camera.intrinsic_matrix)
        image_points = self.camera.intrinsic_matrix.dot(camera_points.T)/z_pos
        #print("image_points_zy",image_points)
        image_points = image_points.T
        #print("+++++++++",z_pos,image_points,"++++++")
        # image=np.array([
        #     (image_points[0][0],image_points[0][1]),
        #     (image_points[1][0],image_points[1][1]),
        #     (image_points[2][0],image_points[2][1]),
        #     (image_points[3][0],image_points[3][1])],dtype= np.float32
        # )
        image=np.array([
            (image_points[0][0]/image_points[0][2],image_points[0][1]/image_points[0][2]),
            (image_points[1][0]/image_points[1][2],image_points[1][1]/image_points[1][2]),
            (image_points[2][0]/image_points[2][2],image_points[2][1]/image_points[2][2]),
            (image_points[3][0]/image_points[3][2],image_points[3][1]/image_points[3][2])],dtype= np.float32
        )
        print("image_points",image)
        model_points = np.array([(250,275,0),(-250,-25,0),(-250,275,0),(250,-25,0)],dtype = np.float32)
      
        #distort = np.array([0.115803,-0.276632,0.000304,0.001818,0],dtype = np.float32)
        distort = np.array([0.153,-0.494,-0.0008,0.0008,0.44],dtype = np.float32)
        (success,rot_vec,trans_vec) = cv2.solvePnP(model_points,image,self.camera.intrinsic_matrix,distort)
        self.camera.calibrate_h = np.concatenate((cv2.Rodrigues(rot_vec)[0],trans_vec),axis=1)
        self.camera.calibrate_h = np.concatenate((self.camera.calibrate_h, np.array([[0,0,0,1]])),axis=0)
        self.camera.calibrate_h = np.linalg.inv(self.camera.calibrate_h)
        #print(self.camera.calibrate_h)

        #dest_image_points = np.array([925,220,375,550,375,220,925,550]).reshape((4,2))
        dest_image_points = np.array([900,210,400,510,400,210,900,510]).reshape((4,2))
        self.camera.homography_matrix = cv2.findHomography(image_points,dest_image_points)[0]
        self.camera.homography_matrix_2 = cv2.findHomography(dest_image_points,image_points)[0]
        print("Homo",self.camera.homography_matrix)
        #print("Homo",self.camera.homography_matrix,self.camera.homography_matrix.dot(np.array([[925],[220],[1]])))
        #320 256 1280-320 256 1280-320 640 320 640
        
        self.status_message = "Calibration - Completed Calibration"
        



    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        self.camera.detectedflag = True
        self.next_state = "idle"
        
        #rospy.sleep(1)

    def line(self):
        #self.rxarm.initialize()
        big_blocks = self.camera.big_blocks.copy()
        small_blocks = self.camera.small_blocks.copy()
        big_colors = self.camera.big_colors.copy()
        small_colors = self.camera.small_colors.copy()
        print("big_blocks:",big_blocks)
        print("small_blocks",small_blocks)
        pos = {'r':0,'o':1,'y':2,'g':3,'b':4,'p':5}
        big_block = big_blocks.copy()
        for (i,point) in enumerate(big_blocks):
            big_block[pos[big_colors[i]]] = point

        for i,point in enumerate(big_block):
            #print(i,point)
            print("point",point)
            offset  = (point[0]-650)>0
            world = self.get_worldframe(point[0],point[1],sign = offset,big=True)
            #world[2] = world[2] - 8
            height = world[2]/60
            #while height>0:
                #print("offset",offset)
                #offset = 0
                #print("point[0],point[1] depth:",self.camera.DepthFrameRaw[int(point[1]),int(point[0])])
            self.grab(world,point[2],sign = offset)
            origin = np.array([120+(42*i)+i,0,10])

            self.grab(origin,0,sign = True)
            #height = height-1
            #world[2] = world[2]-50
                #index = index + 1

        index = 0
        small_block = small_blocks.copy()
        for (i,point) in enumerate(small_blocks):
            small_block[pos[small_colors[i]]] = point

        for i,point in enumerate(small_block):
            offset  = (point[0]-650)>0
            world = self.get_worldframe(point[0],point[1],sign = offset)
            #world[2]  = world[2]/40
            #small_height = world[2]/30
            #while small_height>0:
            self.grab(world,point[2],sign = offset)
            origin = np.array([-340+30*(i),0,10])
            self.grab(origin,-90,sign = False)
            #small_height = small_height-1
            #world[2] = world[2]-25
                #index = index + 1
        
        self.rxarm.sleep()
        self.next_state = "idle"
        pass
    
    def stack(self):
        #self.rxarm.initialize()
        big_blocks = self.camera.big_blocks.copy()
        small_blocks = self.camera.small_blocks.copy()
        big_colors = self.camera.big_colors.copy()
        small_colors = self.camera.small_colors.copy()
        print("big_blocks:",big_blocks)
        print("small_blocks",small_blocks)
        index = 0
        pos = {'r':0,'o':1,'y':2,'g':3,'b':4,'p':5}
        big_block = big_blocks.copy()
        small_block = small_blocks.copy()
        for (i,point) in enumerate(big_blocks):
            big_block[pos[big_colors[i]]] = point

        for i in range(len(big_block)):
            #print(i,point)
            point = big_block[i]
            print("point",point)
            offset  = (point[0]-650)>0
            world = self.get_worldframe(point[0],point[1],sign = offset,big=True)
            world[2] = world[2] - 8
            height = world[2]/50
            index = 0
            #while height>0:
            #print("offset",offset)
            #offset = 0
            #print("point[0],point[1] depth:",self.camera.DepthFrameRaw[int(point[1]),int(point[0])])
            self.grab(world,point[2],sign = offset)
            origin = np.array([150,40,10+36*(i)])
            self.grab(origin,0,sign = True)
            height = height-1
            #world[2] = world[2]-50
                #index = index + 1

        index = 0
        small_block = small_blocks.copy()
        for (i,point) in enumerate(small_blocks):
            small_block[pos[small_colors[i]]] = point

        for i in range(len(small_block)):
            point = small_block[i]
            offset  = (point[0]-650)>0
            world = self.get_worldframe(point[0],point[1],sign = offset)
            print("my world",world)
            #world[2]  = world[2]
            #small_height = world[2]/30
            #while small_height>0:
            self.grab(world,point[2],sign = offset)
            origin = np.array([-200,50,5+25*(i)])
            self.grab(origin,-90,sign = False)
            #small_height = small_height-1
            #world[2] = world[2]-25
                #index = index + 1
        
        self.rxarm.sleep()
        self.next_state = "idle"

    def nstack(self):
    #self.rxarm.initialize()
        big_blocks = self.camera.big_blocks.copy()
        small_blocks = self.camera.small_blocks.copy()
        #big_colors = self.camera.big_colors.copy()
        #small_colors = self.camera.small_colors.copy()
        print("big_blocks:",big_blocks)
        print("small_blocks",small_blocks)
        index = 0
        #pos = {'r':0,'o':1,'y':2,'g':3,'b':4,'p':5}
        #big_block = big_blocks.copy()
        #small_block = small_blocks.copy()
        #for (i,point) in enumerate(big_blocks):
        #    big_block[pos[big_colors[i]]] = point
        for i in range(len(big_blocks)):
            #print(i,point)
            point = big_blocks[i]
            print("point",point)
            offset  = (point[0]-650)>0
            world = self.get_worldframe(point[0],point[1],sign = offset,big=True)
            world[2] = world[2] - 8
            height = world[2]/50
            index = 0
            #while height>0:
            #print("offset",offset)
            #offset = 0
            #print("point[0],point[1] depth:",self.camera.DepthFrameRaw[int(point[1]),int(point[0])])
            self.grab(world,point[2],sign = offset)
            origin = np.array([150,40,10+36*(i)])
            self.grab(origin,0,sign = True)
            height = height-1
            #world[2] = world[2]-50
                #index = index + 1

        # index = 0
        # small_block = small_blocks.copy()
        # for (i,point) in enumerate(small_blocks):
        #     small_block[pos[small_colors[i]]] = point

        for i in range(len(small_blocks)):
            point = small_blocks[i]
            offset  = (point[0]-650)>0
            world = self.get_worldframe(point[0],point[1],sign = offset)
            print("my world",world)
            #world[2]  = world[2]
            #small_height = world[2]/30
            #while small_height>0:
            self.grab(world,point[2],sign = offset)
            origin = np.array([-200,50,5+25*(i)])
            self.grab(origin,-90,sign = False)
            #small_height = small_height-1
            #world[2] = world[2]-25
                #index = index + 1

        self.rxarm.sleep()
        self.next_state = "idle"

    def big_small(self):

        # self.detect_grab_flag = True
        #self.detect()
        big_blocks = self.camera.big_blocks.copy()
        small_blocks = self.camera.small_blocks.copy()
        big_colors = self.camera.big_colors.copy()
        small_colors = self.camera.small_colors.copy()
        print("big_blocks:",big_blocks)
        print("small_blocks",small_blocks)
        #self.initialize_rxarm()
        index = 0
        for i,point in enumerate(big_blocks):
            #print(i,point)
            print("point",point)
            offset  = (point[0]-650)>0
            world = self.get_worldframe(point[0],point[1],sign = offset,big=True)
            world[2] = world[2] - 8
            height = world[2]/50
            while height>0:
                #print("offset",offset)
                #offset = 0
                #print("point[0],point[1] depth:",self.camera.DepthFrameRaw[int(point[1]),int(point[0])])
                self.grab(world,point[2],sign = offset)
                origin = np.array([350-40*(i+index),-20,10])
                self.grab(origin,0,sign = True)
                height = height-1
                world[2] = world[2]-50
                index = index + 1

        index = 0
        for i,point in enumerate(small_blocks):
            offset  = (point[0]-650)>0
            world = self.get_worldframe(point[0],point[1],sign = offset)
            #world[2]  = world[2]
            print("smallHstack",world)
            small_height = world[2]/30
            while small_height>0:
                self.grab(world,point[2],sign = offset)
                origin = np.array([-350+30*(i+index),-20,10])
                self.grab(origin,-90,sign = False)
                small_height = small_height-1
                world[2] = world[2]-25
                index = index + 1
        
        self.rxarm.sleep()
        self.next_state = "idle"
        #rospy.sleep(1)

    def find_joint_ik(self,world_frame,elbow_pose):
        print("ikchecking",world_frame)
        world_x = world_frame[0]
        world_y = world_frame[1]
        world_z = world_frame[2]
        # if world_y<0:
        #     world_z = world_z-60
        # else:
        #     world_z = world_z-25
        l1 = 205.73
        l2 = 200
        l3 = 131+30#131 is 65+66, 20 is at middle of gripper

        x2d = 0
        y2d = 0
        if (elbow_pose==1):
            if self.rxarm.grab == 0: #elbow up, trying to grab
                x2d = pow(world_x*world_x+world_y*world_y,0.5)
                y2d = l3-103.91+world_z#-30
            elif self.rxarm.grab == 1: #elbow up, trying to release
                x2d = pow(world_x*world_x+world_y*world_y,0.5)
                y2d = l3-103.91+world_z#+10
        else:
            print("hi")
            if self.rxarm.grab ==0 :#elbow down trying to grab
                x2d = pow(world_x*world_x+world_y*world_y,0.5)-l3#-23
                y2d = world_z-103.91#-10
            elif self.rxarm.grab ==0 :#elbow down trying to release
                x2d = pow(world_x*world_x+world_y*world_y,0.5)-l3#-23
                y2d = world_z-103.91#+20

        
        tmp = (x2d*x2d+y2d*y2d-l1*l1-l2*l2)/(-2*l1*l2)
        if(tmp<-1):
            tmp = -1
        elif tmp>1:
            tmp=1
            
        if (tmp<=1) and (-1<=tmp):
            th2_hat = np.arccos(tmp)
            th2 = th2_hat-np.arcsin(50/l1)-np.pi/2
        # elif tmp>1:
        #     tmp-=2
        #     th2_hat = np.arccos(tmp)+np.pi
        #     th2 = th2_hat-np.arcsin(50/l1)-np.pi/2
        # elif tmp<-1:
        #     tmp+=2
        #     th2_hat = np.arccos(tmp)+np.pi
        #     th2 = th2_hat-np.arcsin(50/l1)-np.pi/2
        # else:
        #     print("th2 wrong")
        #     return 0
       
        
        try:
            th1_hat = np.arctan2(y2d,x2d)+np.arctan2(l2*np.sin(th2_hat),l1-l2*np.cos(th2_hat))
        except:
            print("th1_hat wrong")
            return 0
        else:
            th1 = -(th1_hat+np.arcsin(50/l1)-np.pi/2)

        if (elbow_pose==1):
            th3 = 3*np.pi/2-(th1_hat+th2_hat)-np.pi
        else:
            th3 = 2*np.pi-(th1_hat+th2_hat)-np.pi

        th0 = np.arctan2(world_y,world_x)-np.pi/2
        if (world_x<0) and (world_y<0):
            th0 = 3*np.pi/2+np.arctan2(world_y,world_x)
        return [th0,th1,th2,th3,0]
    

    def get_worldframe(self,p_x,p_y,sign=True,stack=False,line = False,big=False):
        area = self.camera.DepthFrameRaw[int(p_y-15):int(p_y+15),int(p_x-15):int(p_x+15)]
        z = area.min()
        # r,c = np.where(area==z)
        # print(r,c)
        # r_mean = r.mean()-15
        # c_mean = c.mean()-15
        # p_x = int(p_x + r_mean)
        # p_y = int(p_y + c_mean)
        self.camera.last_click[0] = p_x
        self.camera.last_click[1] = p_y
        k_inv = np.linalg.inv(self.camera.intrinsic_matrix)
        pos = np.array([self.camera.last_click[0],self.camera.last_click[1],1])
        p_new = np.linalg.solve(self.camera.homography_matrix,pos)
        p_new = p_new/p_new[2]
        #z = self.camera.DepthFrameRaw[self.camera.last_click[1]-15:self.camera.last_click[1]+15][self.camera.last_click[0]-15:self.camera.last_click[0]+15].min()
        came_frame = z*np.matmul(k_inv,np.array([p_new[0],p_new[1],1]))
        came_frame = np.array([came_frame[0],came_frame[1],came_frame[2],1]).T
        world_frame = np.matmul(self.camera.calibrate_h,came_frame)
        world_frame = world_frame/world_frame[3]
        off = -10-world_frame[1]*20/450
        world_frame[2] = max(0,world_frame[2]+off)

        if(sign):
            world_frame[1] = world_frame[1]+15
            world_frame[0] = world_frame[0]+world_frame[0]/10-10
        else:
            print("origin_x",world_frame[0])
            world_frame[1] = world_frame[1]+(world_frame[1]/20)
            world_frame[0] = world_frame[0]+world_frame[0]/8
            print("changedx",world_frame[0])
            
        # if(stack):
        #     print("stacked!")
        #     if self.rxarm.grab==1:
        #         print("lowe")
        #         world_frame[2] = max(0,world_frame[2]-15)
        #     else:
        #         print("lowe")
        #         world_frame[2] = max(0,world_frame[2]-10)
        
        if(big):
            print("stacked!")
            if self.rxarm.grab==1:
                print("lowe")
                world_frame[2] = max(10,world_frame[2]-15)
            else:
                print("lowe")
                world_frame[2] = min(20,world_frame[2]-25)
        else:
            if self.rxarm.grab==1:
                print("lowe")
                world_frame[2] = max(3,world_frame[2]-20)
            else:
                print("lowe")
                world_frame[2] = max(6,world_frame[2]-30)
        return world_frame

    def grab(self,world_frame,angle=0,sign=True,stack=False):
        #self.camera.new_click = True
        # else:
        #     world_frame[0]+world_frame/10
        #     world_frame[1]-world_frame[1]/20
        print("world_frmae::",world_frame[0],world_frame[1])
        next_joint_ik = self.find_joint_ik(world_frame,1)
        # mid = world_frame
        # sign = 1*(world_frame>0)-1*(world_frame<0)
        # mid = mid-sign*10
        # mid_joint = self.find_joint_ik(mid,1)
        
        if next_joint_ik == 0:
            # mid = world_frame
            # mid[0],mid[1] = world_frame[0]-20*sign[0],world_frame[1]-20*sign[1]
            # mid_joint = self.find_joint_ik(mid,1)
            next_joint_ik = self.find_joint_ik(world_frame,2)
        else:
            print(next_joint_ik)
           # self.rxarm.set_joint_positions(joint_ik,moving_time=2.0,accel_time=0.5,blocking=True)           
        if next_joint_ik == 0:
            print("unreachable")
            return 0
        
        #theta_5 = np.abs(angle*np.pi/180)-(np.pi/2-np.abs(next_joint_ik[0]))
        if next_joint_ik[0]<0:
            theta_5 = np.abs(angle*np.pi/180)-(np.pi/2-np.abs(next_joint_ik[0]))
        else:
            theta_5 = -np.abs(np.pi/2+angle*np.pi/180)+(np.pi/2-next_joint_ik[0])
            
        print("next_joint_ik",next_joint_ik)
        theta_5 = -theta_5
        if(stack and self.rxarm.grab==1):
            print("stacked and theta is 0")
            theta_5 = 0
        current_joint_ik = self.rxarm.get_positions()
        #high_pos = world_frame.copy()
        #high_pos[2]=high_pos[2]+50
        #high_ik = self.find_joint_ik(high_pos,1)
        #self.rxarm.set_joint_positions(high_ik,moving_time=1.0,accel_time=0.5,blocking=True)

        world_frame[2] = world_frame[2]+50
        high = world_frame.copy()
        high[2] = high[2]+50
        high_pos = self.find_joint_ik(high,1)
        high_pos[-1] = theta_5

        # wayplan =  [
             # [current_joint_ik[0],     0,    current_joint_ik[2],  current_joint_ik[3], current_joint_ik[-1]],
              #[current_joint_ik[0],     0,    0,      0.0,     theta_5],
            #   [0                  ,     0,    0,        0,       0],
            #     high_pos,
            #   [next_joint_ik[0],     0,    0,      0.0,     theta_5]]
            #   [next_joint_ik[0],     0,    0,      next_joint_ik[-2],     theta_5]]
        
        # for wp in wayplan:
        #     if(isinstance(next_joint_ik,int)): 
        #         self.rxarm.sleep()
        #         return 0
        #     self.rxarm.set_joint_positions(wp,moving_time=2.0,accel_time=0.5,blocking=True)
        
        self.rxarm.set_joint_positions(high_pos,moving_time=2,accel_time=0.5,blocking=True)
        
        for i in range(2):
            world_frame[2] = world_frame[2] - i*15
            next_joint_ik = self.find_joint_ik(world_frame,1)
            next_joint_ik[-1] = theta_5
            if(isinstance(next_joint_ik,int)): 
                return 0
            self.rxarm.set_joint_positions(next_joint_ik,moving_time=2,accel_time=0.5,blocking=True)
        
        
        #self.rxarm.set_joint_positions(next_joint_ik,moving_time=2.0,accel_time=0.5,blocking=True)
        if self.rxarm.grab==0:
            self.rxarm.close_gripper()
            self.rxarm.grab=1
            #high = [next_joint_ik[0],     0,    0,      0.0,     theta_5]
            self.rxarm.set_joint_positions(high_pos,moving_time=2,accel_time=0.5,blocking=True)
        else:
            self.rxarm.open_gripper()
            self.rxarm.grab=0
            self.rxarm.set_joint_positions(high_pos,moving_time=2,accel_time=0.5,blocking=True)

        return 0
    
    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            rospy.sleep(5)
        self.next_state = "idle"

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            rospy.sleep(0.05)