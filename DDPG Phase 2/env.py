#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
world = False
if world:
    from respawnGoal_custom_worlds import Respawn
else:
    from respawnGoal import Respawn
import copy
target_not_movable = False

class Env():
    def __init__(self, action_dim=2):
        self.max_action = np.array([0.26, 1.82])
        self.min_action = np.array([-0.26, -1.82])
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.past_position = None
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        self.past_distance = 0
        self.stopped = 0
        self.action_dim = action_dim
        #Keys CTRL + c will stop script
        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        #you can stop turtlebot by publishing an empty Twist
        #message
        rospy.loginfo("Stopping TurtleBot")
        self.pub_cmd_vel.publish(Twist())
        rospy.sleep(1)

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        self.past_distance = goal_distance

        return goal_distance

    def getOdometry(self, odom):
        self.past_position = copy.deepcopy(self.position)
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        #print 'yaw', yaw
        #print 'gA', goal_angle

        heading = goal_angle - yaw
        #print 'heading', heading
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 3)

    def getState(self, scan, past_action):
        scan_range = []
        heading = self.heading
        min_range = 0.22 #0.136
        done = False
        goal = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf') or scan.ranges[i] == float('inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]) or scan.ranges[i] == float('nan'):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])
        # Calculate interval
        if len(scan_range) != 360:
            interval = 360 / len(scan_range)
            # Create empty array for scaled values
            scaled_array = np.zeros(360)

            # Interpolate missing values
            for i in range(360):
                original_index = int(i / interval)
                if original_index == len(scan_range) - 1:
                    scaled_array[i] = scan_range[original_index]
                else:
                    lower_value = scan_range[original_index]
                    upper_value = scan_range[original_index + 1]
                    fraction = i / interval - original_index
                    scaled_array[i] = lower_value + fraction * (upper_value - lower_value)
            scan_range = scaled_array

        # if len(scan_range) != 360:
        #     lidar_220 = scan_range  
        #     # Define the new resolution (360 points)
        #     new_resolution = 360
        #     # Upsample using linear interpolation
        #     lidar_360 = np.interp(np.linspace(0, 1, new_resolution), np.linspace(0, 1, len(lidar_220)), lidar_220)
        #     scan_range = lidar_360

        if min_range > min(scan_range) > 0.205:
            done = True
        # scan_range = scan_range[::36]
        scan_range = [x / 3.5 for x in scan_range]

        # Past actions were being appended for some reason
        for pa in past_action:
            scan_range.append(pa)
        # print(type(scan_range))
        # print(scan_range)

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        # current_distance = self.getGoalDistace()
        if current_distance < 0.30:
            self.get_goalbox = True
            goal = True
        # print("Scanrange", len(scan_range))

        return scan_range + [heading, current_distance], done, goal

    def setReward(self, state, done):
        current_distance = state[-1]
        heading = state[-2]
        scan_r = state[:-3]
        #print('cur:', current_distance, self.past_distance)

        
        distance_rate = (self.past_distance - current_distance) 
        # angle_reward = math.pi - abs(heading)   OLD
        angle_reward = math.pi/2 - abs(heading)
        # -2 for removing the past actions from the distance of obstacles array, if the dimensions of input_actions is not 2 please change this to -x where x is dimensions of action
        avoidance_rate = self.calculate_obstacle_avoidance_reward(scan_r[:(len(scan_r)-2)])
        
        # if distance_rate  0:
        # print("Distance Reward =", 500.*distance_rate, "\nAngle Reward=", 3.*angle_reward, "\nAvoidance=", -1000*avoidance_rate)
        # reward = 500.*distance_rate + 3.*angle_reward + (-100.*avoidance_rate)   OLD
        reward = 500.*distance_rate + 3.*angle_reward - (1000.*avoidance_rate)

        # if distance_rate == 0:
        #     reward = 0.
        # For -ve distance rate a constant negative reward was being given, changed it to varying according to speed away from goal
        if distance_rate <= 0:
            # reward = 500.*distance_rate + (-100.*avoidance_rate)  OLD
            reward = -15 + 3.*angle_reward - (1000.*avoidance_rate)
            # reward = 0.
        
        #print('d', 500*distance_rate)
        
        self.past_distance = current_distance

        a, b, c, d = float('{0:.3f}'.format(self.position.x)), float('{0:.3f}'.format(self.past_position.x)), float('{0:.3f}'.format(self.position.y)), float('{0:.3f}'.format(self.past_position.y))
        if a == b and c == d:
            # rospy.loginfo('\n<<<<<Stopped>>>>>\n')
            # print('\n' + str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d) + '\n')
            self.stopped += 1
            if self.stopped == 20:
                rospy.loginfo('Robot is in the same 20 times in a row')
                self.stopped = 0
                done = True
        else:
            # rospy.loginfo('\n>>>>> not stopped>>>>>\n')
            self.stopped = 0

        if done:
            rospy.loginfo("Collision!!")
            # reward = -500.
            reward = -1000
            self.pub_cmd_vel.publish(Twist())
            # print("The LIDAR values are:", state)

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            # reward = 500.
            # reward = 1000.
            reward = 2000.
            self.pub_cmd_vel.publish(Twist())
            if world:
                self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True, running=True)
                if target_not_movable:
                    self.reset()
            else:
                self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward, done

    def calculate_obstacle_avoidance_reward(self, scan_r):
    # Hypothetical lidar reading for obstacle detection
        lidar_reading = scan_r

    # Threshold distance for considering obstacle proximity
        obstacle_threshold = 0.09  # Adjust as needed
    
    # If lidar reading indicates obstacle proximity, penalize
        if min(lidar_reading) < obstacle_threshold:
        # Penalize based on the proximity to the obstacle
            avoidance_rate = (obstacle_threshold - min(lidar_reading))  # OLD AVOIDANCE RATE
            # avoidance_rate = 0.03
        else:
            avoidance_rate = 0  # No penalty if no obstacles
    
        return avoidance_rate


    def step(self, action, past_action):
        linear_vel = action[0]
        ang_vel = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done, goal = self.getState(data, past_action)
        reward, done = self.setReward(state, done)

        distance_traveled = math.sqrt((self.position.x - self.past_position.x)**2 + (self.position.y - self.past_position.y)**2)

        return np.asarray(state), reward, done, goal, distance_traveled

    def reset(self):
        #print('aqui2_____________---')
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False
        else:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)

        self.goal_distance = self.getGoalDistace()
        state, _ , _= self.getState(data, [0,0])

        return np.asarray(state)
