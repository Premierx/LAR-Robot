from __future__ import print_function
from robolab_turtlebot import Turtlebot, Rate, get_time, sleep
import cv2
import numpy as np

WINDOW = 'obraz'

 
#Steering tolerance to be set and determined by testing. Stop distance is flawed. Determines closest vector.
#Min area is too large I think. might interfere with object detection at larger distances.
SCREEN_CENTER_X = 319
STEERING_TOLERANCE = 40
STOP_DISTANCE = 0.7
MIN_AREA = 400
#MIN_AREA =


SEARCH_STOP = 1.0
LAST_GLOBAL_TIME = 0.0
touch = [0, 0]  # [bumper_id, state]
#Green is okay but has issues with over exposed light conditions.
LOWER_GREEN  = np.array([35,  60,  60])
UPPER_GREEN  = np.array([90,  255, 255])

#TODO: Purple either includes noise or underdetects some areas. Works but could be better. It has issues at closer rather than long range.
#Shadows are purple :)
LOWER_PURPLE = np.array([110,70,40]) 
UPPER_PURPLE = np.array([150,255,220])

#This is simply a matrix of ones used to fill holes in the mask in the hope of fixing the short range purple detection problem. Doesnt really work.
KERNEL = np.ones((9, 9), np.uint8)

#Just bools that determine the progress of the code
begin_parked = True
ball_found   = False
find_garage  = False
garage_found = False
searching = False
measure_time = True
time_measured= False
bumper_pause = False

def bumper_cb(msg):
    global touch
    touch[0] = msg.bumper   # 0:LEFT, 1:CENTER, 2:RIGHT
    touch[1] = msg.state    # 0:RELEASED, 1:PRESSED

def preprocess(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)
    #TODO
    hsv = cv2.blur(hsv, (4, 4)) 
    #hsv = cv2.GaussianBlur(hsv, (9, 9), 0)  #Noise reduction to test
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # CLAHE is contrast enhancement. Should help with lighting...maybe :D
    #hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    return hsv

# Function for getting the mask of the chosen color. Output is using the KERNEL matrix to close the holes. 
def get_mask(hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
   
    return mask #cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

# Function that simply returns the index of the largest green component. 
def largest_component(out):
    if out[0] < 2:
        return None
    valid = [i for i in range(1, out[0]) if out[2][i][4] > MIN_AREA] # This is the MIN_AREA filter. Should tweak this. Maybe separate constant for green and purple?
    if not valid:
        return None

    return max(valid, key=lambda i: out[2][i][4])

# Aaaand the same function for the purple pilons. It gets all blobs bigger than some area, sorts them and returns the biggest two.
def find_two_largest(out):
    if out[0] < 3:
        return None
    valid = [i for i in range(1, out[0]) if out[2][i][4] > MIN_AREA] #TODO: Make the constant smaller.
    if len(valid) < 2:
        return None

    return sorted(valid, key=lambda i: out[2][i][4], reverse=True)[:2]

# This checks the direction of the object of interest. This and the drive function can be both integrated into the P regulator.
def check_direction(target_x):
    offset = target_x - SCREEN_CENTER_X
    if offset > STEERING_TOLERANCE:  return int(((-1)*offset)/(320-STEERING_TOLERANCE)) #Output range is (STEERING_TOLERANCE/100, 319-STEERING_TOLERANCE/100)
    if offset < -STEERING_TOLERANCE: return  int(offset/(320-STEERING_TOLERANCE))

    return 0
# This is b_llshit. The robot shakes like a crackhead in withdrawal.
#TODO: At least a P regulator.
def drive(direction):
    if direction == 0:
        turtle.cmd_velocity(linear=0.2)
    else:
        turtle.cmd_velocity(linear= 0.2, angular= direction * 0.4)

# Do I have to comment this?
def stop():
    turtle.cmd_velocity(linear=0.0, angular=0.0)

def get_out():
    dist = float(get_depth_at(360, 150))#This was moved to the right a little so the bot doesnt scrape the garage
    if (dist<1):
        turtle.cmd_velocity(angular=0.5)
        return 0
    if (dist>=1.0):
        return 1

# This should trigger only some part of the camera rather than whole camera to optimize the process. This is Claude AI. Just a proof of concept. 
# Id pursue this or something simmilar. This "opens" a 100x100 window around the point of interest. Either around the centre of the green ball or the larger of the two pilons.
# For the ball its okay but for the garage the robot is navigated in the middle of the pilons. Id call this function again after "finding the garage" and call some sort of a parking function with smaller distance requirements than the 0.7 meters for the two pilons in order to park in between them.
def get_depth_at(centroid_x, centroid_y):
    pc = turtle.get_point_cloud()
    if pc is None:
        return None

    cx, cy = int(centroid_x), int(centroid_y)
    h, w = pc.shape[:2]
    x0, x1 = max(0, cx-30), min(w, cx+30)
    y0, y1 = max(0, cy-30), min(h, cy+30)
    region = pc[y0:y1, x0:x1, 2]
    valid = region[(region > 0.1) & (region < 5.0)]
    if len(valid) == 0:
        return None

    return float(np.median(valid))
 
#Thresholds and other functions that are no longer used have been removed.
#For better readability some repetetive tasks have been made into functions.
def main():
    global ball_found, find_garage, garage_found, begin_parked, searching, measure_time,bumper_pause, time_measured
    global LAST_GLOBAL_TIME
    rate = Rate(10)
    cv2.namedWindow(WINDOW)
    turtle.wait_for_rgb_image()
    turtle.register_bumper_event_cb(bumper_cb)

    while not turtle.is_shutting_down():
        key = cv2.waitKey(1) & 0xFF
        if touch[1] == 1 and bumper_pause == False:
            bumper_pause = True
            touch[1] = 0
            
        if key == ord('r'):
            bumper_pause = False    

        if bumper_pause:
            stop()
            continue

        image_rgb = turtle.get_rgb_image()
        hsv = preprocess(image_rgb)
        cv2.imshow(WINDOW, image_rgb)
        if key == ord('q'):
            break
        
        if begin_parked:
            get_out_status=get_out()
            if get_out_status==1:
                if measure_time:    
                    LAST_GLOBAL_TIME = get_time()
                    time_measured = True
                    measure_time = False
                if time_measured:
                    drive_fwd_t = 1.0
                    time_threshold = get_time() - drive_fwd_t 
                    if (time_threshold < LAST_GLOBAL_TIME):
                        drive(0)
                    elif (time_threshold > LAST_GLOBAL_TIME):
                        stop()
                        begin_parked = False
                        time_measured= False
                        measure_time = True
                        
                    else:
                        stop()
                        begin_parked = False
                else:
                    print("Time has not been measured.")
                    continue
              
        # The robot begins by looking for the green ball.
        elif not ball_found and not begin_parked:
            mask = get_mask(hsv, LOWER_GREEN, UPPER_GREEN)
            cv2.imshow('mask', mask)
            out = cv2.connectedComponentsWithStats(mask)
            idx = largest_component(out)
            
            #TODO: Maybe more sophisticated search algorithm than, well... turning around? 
            if idx is None and searching:
                #print("Ball not found... Searching....")
                turtle.cmd_velocity(angular=0.3)
                
                continue

            cx, cy = int(out[3][idx][0]), int(out[3][idx][1])
            direction = check_direction(cx)

            if direction != 0:
                drive(direction)
                rate.sleep()
                continue

            # Driving until the set distance for the ball is reached.
            depth = get_depth_at(cx, cy)

            print(f"Distance of the ball: {depth:.2f} m" if depth else "Ball distance unavailable.")

            # Just determining when to stop.
            if depth is not None and depth <= STOP_DISTANCE:
                stop()
                ball_found  = True
                find_garage = True
                print("Set distance from the ball has been reached. Begin the search for the garage.")

            else:
                drive(0)
                rate.sleep()

        # Looking for the garage.
        elif find_garage and not garage_found:
            mask = get_mask(hsv, LOWER_PURPLE, UPPER_PURPLE)
            cv2.imshow('mask', mask)
            out = cv2.connectedComponentsWithStats(mask)
            indices = find_two_largest(out)

            #TODO: The same as for the ball. This search pattern is not redundant enough.
            if indices is None and searching:
                #print("Garage pilons not found... Searching ...")
                turtle.cmd_velocity(angular=0.3)
                
                continue
            
            #Determining where to go. Ideally in the middle of the two pilons.
            x1 = int(out[3][indices[0]][0])
            x2 = int(out[3][indices[1]][0])
            y1 = int(out[3][indices[0]][1])
            midpoint_x = (x1 + x2) // 2
            direction = check_direction(midpoint_x)

            if direction != 0:
                drive(direction)
                rate.sleep()

                continue
            
            # Distance to the closest pilon...
            depth = get_depth_at(x1, y1)
            print(f"Distance from the closest pilon:  {depth:.2f} m" if depth else "Distance from the pilon not available.")
 
            if depth is not None and depth <= STOP_DISTANCE:
                stop()
                garage_found = True
                print("Garage found. Task COMPLETED.")

            else:
                drive(0)
                rate.sleep()

        elif garage_found:
            stop()
        #TODO: Id add the regulator for the driving enhancement and something that follows after the garage has been found. A parking function one might say.

if __name__ == '__main__':
    turtle = Turtlebot(rgb=True, depth=True, pc=True)
    main()
