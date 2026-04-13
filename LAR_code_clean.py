from __future__ import print_function
from robolab_turtlebot import Turtlebot, Rate, get_time, sleep
import cv2
import numpy as np
import math

# Window settings
WINDOW = 'obraz'
SCREEN_CENTER_X    = 319

# Driving constants
STEERING_TOLERANCE = 20
STOP_DISTANCE  = 0.5
STOP_DISTANCE_P = 0.7
TURN_RADIUS = STOP_DISTANCE - 0.15

# Circle constants
TURN_RIGHT = 60.0
CIRCLE_ANGLE  = 345.0  # degrees to sweep around ball
BALL_CLOSE_THRESHOLD = 0.75

# Circle P-controller settings
KP_HEADING    = 0.1    # heading correction gain
KD_DIST       = 0.4    # radial distance correction gain
CIRCLE_LINEAR = 0.15   # forward speed while circling

#Garage specifications
GARAGE_WIDTH = 52
GARAGE_DEPTH = 44
SCREEN_CENTER_Y = 239
PARKING_DISTANCE = 0.45

# HSV constraints and mask filters
LOWER_GREEN = np.array([35,  60,  60])
UPPER_GREEN = np.array([90, 255, 255])
LOWER_PURPLE = np.array([110, 70,  40])
UPPER_PURPLE = np.array([150, 255, 220])
MIN_AREA = 400
KERNEL = np.ones((9, 9), np.uint8)

# Odometry
start_position = []
ball_position  = []

# get_out() time values
garage_left_p_time = 0.0
garage_right_p_time = 0.0
pilon_time_diff = 0.0
start_centre_time = 0.0

# Code progression booleans
front_clear = False
front_obstructed = False
got_start_park_time = False
time_diff_found = False
get_start_time = True
begin_parked = True 
ball_found   = False 
find_garage  = False 
garage_found = False

turning_right          = False  # True during the initial 90° turn after finding the ball
circling            = False
get_ball_location   = True
turn_60_start_theta = None

# Circle controller state
ball_world_x    = 0.0
ball_world_y    = 0.0
circle_swept    = 0.0
circle_prev_phi = 0.0
circle_prev_phi = 0.0

# Bumper states
touch = [0, 0]
button = 0

# ================================================================ #
#   Bumber callback:
#   msg.bumper stores the id of bumper 0:LEFT, 1:CENTER, 2:RIGHT
#   msg.state stores the event 0:RELEASED, 1:PRESSED
# ================================================================ #
def bumper_cb(msg):
    global touch
    touch[0] = msg.bumper
    touch[1] = msg.state

# ================================================================ #
#   Button callback
# ================================================================ #
def button_cb(msg):
    global button
    """Button callback."""
    button = msg.state
    """Stop musi bzt tadz"""

# ================================================================ #
#   Function to convert RGB to HSV and blur the result.
# ================================================================ #
def preprocess(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)
    hsv = cv2.blur(hsv, (4, 4))
    return hsv

# ================================================================ #
#   Filtering the pixels within HSV upper and lowe bounds.
# ================================================================ #
def get_mask(hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
    return mask  

# ================================================================ #
#   Sorting the areas by size and filtering the small ones.
#   Returning the largest area.
# ================================================================ #
def largest_component(out):
    if out[0] < 2:
        return None
    valid = [i for i in range(1, out[0]) if out[2][i][4] > MIN_AREA]
    if not valid:
        return None
    return max(valid, key=lambda i: out[2][i][4])

# ================================================================ #
#   Sorting the areas by size and filtering the small ones.
#   Returning the two largest areas.
# ================================================================ #
def find_two_largest(out):
    if out[0] < 3:
        return None
    valid = [i for i in range(1, out[0]) if out[2][i][4] > MIN_AREA]  
    if len(valid) < 2:
        return None
    return sorted(valid, key=lambda i: out[2][i][4], reverse=True)[:2]

# ================================================================ #
#   Checking the direction by x-coordinate difference from centre.
# ================================================================ #
def check_direction(target_x):
    offset = target_x - SCREEN_CENTER_X
    if offset > STEERING_TOLERANCE:  return -1
    if offset < -STEERING_TOLERANCE: return  1
    return 0

# ================================================================ #
#   Commanding the robots driving speed and direction of travel.
# ================================================================ #
def drive(direction, slow):
    if slow == 2:
        if direction == 0:
            turtle.cmd_velocity(linear=0.05)
        else:
            turtle.cmd_velocity(linear=0.05, angular=direction * 0.15)
    elif slow == 1:
        if direction == 0:
            turtle.cmd_velocity(linear=0.1)
        else:
            turtle.cmd_velocity(linear=0.1, angular=direction * 0.25)
    else:
        if direction == 0:
            turtle.cmd_velocity(linear=0.2)
        else:
            turtle.cmd_velocity(linear=0.2, angular=direction * 0.15)

# ================================================================ #
#   Stopping the robot.
# ================================================================ #
def stop():
    turtle.cmd_velocity(linear=0.0, angular=0.0)

# ================================================================ #
#   Function to park out the robot when started inside the garage.
#   Pixel locations for get_depth_at() were determined by testing.
# ================================================================ #
def get_out():
    global garage_left_p_time, garage_right_p_time, pilon_time_diff, start_centre_time
    global front_clear, front_obstructed, got_start_park_time, time_diff_found, get_start_time
    turn_speed = 0.5
    
    if front_clear == False and front_obstructed == False:
        dist = get_depth_at(319, 150)
        if dist is not None and dist > 1.0: front_clear = True
        if dist is not None and dist <= 1.0: front_obstructed = True
    
    if front_obstructed:
        centre_right_dist = get_depth_at(400,150)
        if centre_right_dist is not None and centre_right_dist < 1.0:
            turtle.cmd_velocity(angular = turn_speed / 1.5)
            return 0

    elif front_clear and not time_diff_found:
        if not got_start_park_time:
            left_p_dist = get_depth_at(100,150)

            if left_p_dist is not None and left_p_dist > 1.0:
                turtle.cmd_velocity(angular = turn_speed)
                return 0

            if left_p_dist is not None and left_p_dist <= 1.0:
                garage_left_p_time = get_time()
                got_start_park_time = True
                return 0
            
        else:
            right_p_dist = get_depth_at(520,150)
            if right_p_dist is not None and right_p_dist  > 1.0:
                turtle.cmd_velocity(angular = -turn_speed / 2)
                return 0
            
            if  right_p_dist is not None and right_p_dist <= 1.0:
                garage_right_p_time = get_time()
                pilon_time_diff = garage_right_p_time - garage_left_p_time
                time_diff_found = True
                return 0
            
    elif time_diff_found:
        if get_start_time:
            start_centre_time = get_time()
            get_start_time = False

        if get_time() - start_centre_time < pilon_time_diff / 2:
            turtle.cmd_velocity(angular = turn_speed / 2)
            return 0
        
        else:
            return 1
        
    return 1

# ================================================================ #
#   Parking function for when the pylon is near the garage.
# ================================================================ #
def emergency_park():
    global garage_found
    start = math.degrees(turtle.get_odometry()[2])
    rate = Rate(10)
    while True:
        current = math.degrees(turtle.get_odometry()[2])
        if abs(angle_diff(start, current)) < TURN_RIGHT - 5:
            turtle.cmd_velocity(angular= -0.2)
        else:        
            depth = get_depth_at(SCREEN_CENTER_X, SCREEN_CENTER_Y+50)

            if depth is not None and depth <= PARKING_DISTANCE:
                stop()
                garage_found = True
                print("Parking finished")
                break
            else:
                drive(0,1)
                rate.sleep()

# ================================================================ #
#   Returns median value of depth around the point of interest.
#   This reduces noise and provides stable return value.
# ================================================================ #
def get_depth_at(centroid_x, centroid_y):
    pc = turtle.get_point_cloud()
    if pc is None:
        return None

    cx, cy = int(centroid_x), int(centroid_y)
    h, w = pc.shape[:2]
    r = 25 # Half size of the median value square
    x0, x1 = max(0, cx - r), min(w, cx + r)
    y0, y1 = max(0, cy - r), min(h, cy + r)
    region = pc[y0:y1, x0:x1, 2]
    valid = region[(region > 0.1) & (region < 5.0)] # Ignoring values smaller than 0.1m and larger than 5.0m
    if len(valid) == 0:
        return None
    
    return float(np.median(valid))

# ================================================================ #
#   Calculation of the angle difference.
# ================================================================ #
def angle_diff(a, b):
    diff = (b - a) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff

# ================================================================ #
#   Returns (x,y) world coordinates of an object set distance in front of the robot.
#   cos/sin of theta already encode direction for all quadrants,
#   so no need to handle positive/negative x,y separately.
#   This reduces noise and provides stable return value.
# ================================================================ #
def world_coord(odometry, distance):
    rx, ry, rtheta = odometry[0], odometry[1], odometry[2]
    bx = rx + distance * math.cos(rtheta)
    by = ry + distance * math.sin(rtheta)
    #print(f"Ball world position: ({bx:.3f}, {by:.3f})")
    return bx, by


# - - - Parking functions - - -
def garage_center_distance(l1, l2):
    return (math.pow(l2, 2) - math.pow(l1, 2)) / (2 * GARAGE_WIDTH)

def garage_parking_distance(l1, l2):
    return GARAGE_DEPTH / 2 + math.sqrt(math.pow(l1, 2) - math.pow(
        (math.pow(l1, 2) - math.pow(l2, 2) + math.pow(GARAGE_WIDTH, 2)) / (-2 * GARAGE_WIDTH), 2))

def garage_parking_angle(l1, l2):
    return math.atan2(garage_center_distance(l1, l2), (garage_parking_distance(l1, l2) - GARAGE_DEPTH / 2))

def get_angle(l1, l2, side):
    return side*garage_parking_angle(l1, l2) - math.atan2( (side*garage_center_distance(l1,l2) - GARAGE_WIDTH/2) , (garage_parking_distance(l1, l2)-GARAGE_DEPTH/2) )
# ^ ^ ^ Parking functions ^ ^ ^

# ================================================================ #
#   Main parking function
#   STATES: 0: Stop, 1: Turning to center, 2: Driving to center, 3: Turning to garage, 4: Driving to garage
# ================================================================ #
def parking(l1, l2):
    global touch
    rate = Rate(10)
    state = 1
    start_time = get_time()
    
    side = math.copysign(1, garage_parking_angle(l1, l2))  # 1: Left of garage (has to turn right first) ; -1: Right
    steering_angle = math.pi/2-side*garage_parking_angle(l1, l2)
    center_distance = garage_center_distance(l1,l2)
    turtle.reset_odometry()

    while touch[1] == 0 and not turtle.is_shutting_down():
        if touch[1] == 1:  # Finding out about the bump
            turtle.cmd_velocity(linear=0, angular=0)
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'): break

        if state == 1:  # Turning to center
            turned = turtle.get_odometry()[2]
            print(f"turned{turned},    `   steering_angle {steering_angle}")
            if -side * turned >= steering_angle:
                state = 2
                turtle.reset_odometry()
                
            else:
                turtle.cmd_velocity(linear = 0, angular = -side * 0.3)

        elif state == 2:  # Driving to center
            curr_pos = turtle.get_odometry()
            if curr_pos[0] >= side * center_distance / 100:
                state = 3
                turtle.reset_odometry()
                
            else:
                if curr_pos[2] > 0: turtle.cmd_velocity(linear=0.1, angular=-0.2)
                elif curr_pos[2] < 0: turtle.cmd_velocity(linear=0.1, angular=0.2)
                else: turtle.cmd_velocity(linear=0.1, angular=0.0)
                rate.sleep()

        elif state == 3:  # Turning to garage
            turned = abs(turtle.get_odometry()[2])
            if turned >= math.pi / 2:
                state = 4
                turtle.reset_odometry()
                
            else:
                turtle.cmd_velocity(angular=side * 0.3)

        elif state == 4:  # Driving to garage
            depth = get_depth_at(SCREEN_CENTER_X, SCREEN_CENTER_Y + 50)

            if depth is not None and depth <= PARKING_DISTANCE:
                stop()
                garage_found = True
                print("Parking finished")
                state = 0
                
            else:
                if turtle.get_odometry()[2] > 0: turtle.cmd_velocity(linear = 0.1, angular = -0.2)
                elif turtle.get_odometry()[2] < 0: turtle.cmd_velocity(linear = 0.1, angular = 0.2)
                else: turtle.cmd_velocity(linear = 0.1, angular = 0.0)
                rate.sleep()

        if state == 0:
            return 0

# For better readability some repetitive tasks have been made into functions.
def main():
    global ball_found, find_garage, garage_found, begin_parked
    global turning_right, circling, turn_60_start_theta
    global start_position, get_ball_location, touch, button
    global ball_world_x, ball_world_y, circle_swept, circle_prev_phi
    global garage_left_p_time, garage_right_p_time, pilon_time_diff, start_centre_time, front_clear, front_obstructed, got_start_park_time, time_diff_found,get_start_time

    rate = Rate(10)
    cv2.namedWindow(WINDOW)
    turtle.wait_for_rgb_image()
    turtle.wait_for_odometry()
    turtle.register_button_event_cb(button_cb)    
    turtle.register_bumper_event_cb(bumper_cb)

    while touch[1] == 0 and not turtle.is_shutting_down():
        image_rgb = turtle.get_rgb_image()
        hsv = preprocess(image_rgb)
        cv2.putText(image_rgb, "Press Q to kill program.", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(200, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow(WINDOW, image_rgb)
        
        if touch[1] == 1: # Finding out about the bump
            turtle.cmd_velocity(linear=0, angular=0)
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # ------------------------------------------------------------------ #
        #  Phase 0: Drive out of starting position                            #
        # ------------------------------------------------------------------ #
        if begin_parked:
            start_position = turtle.get_odometry()
            print(start_position)
            get_out_status = get_out()
            if get_out_status == 1:
                t = get_time()
                while get_time() - t < 1:
                    drive(0, 0)
                begin_parked = False
        # ------------------------------------------------------------------ #
        #  Phase 1: Find and approach the green ball                          #
        # ------------------------------------------------------------------ #
        elif not ball_found and not begin_parked and not circling and not turning_right:
            mask = get_mask(hsv, LOWER_GREEN, UPPER_GREEN)
            cv2.imshow('mask', mask)
            out = cv2.connectedComponentsWithStats(mask)
            idx = largest_component(out)

            if idx is None:
                turtle.cmd_velocity(angular=0.3)
                continue

            cx, cy = int(out[3][idx][0]), int(out[3][idx][1])
            direction = check_direction(cx)
            depth = get_depth_at(cx, cy)

            if depth is not None and depth <= 1.2 and depth > STOP_DISTANCE:
                slow = 1
            else:
                slow = 0

            if depth is not None and depth <= STOP_DISTANCE:
                stop()
                ball_found = True
                turning_right = True

                # Capture ball world position right when we stop
                position = turtle.get_odometry()

                if start_position is not None:
                    dist_from_start = math.sqrt((position[0] - start_position[0])**2 + (position[1] - start_position[1])**2)
                    ball_close = dist_from_start < BALL_CLOSE_THRESHOLD

                ball_world_x, ball_world_y = world_coord(position, TURN_RADIUS)
                turn_60_start_theta = math.degrees(position[2])
                print(f"Ball reached. Starting 90° right turn.")
                continue

            drive(direction, slow)
            rate.sleep()

        # ------------------------------------------------------------------ #
        #  Phase 2: Turn 60° right so we are tangent to the circle            #
        # ------------------------------------------------------------------ #
        elif turning_right and not circling and not find_garage:
            current_theta = math.degrees(turtle.get_odometry()[2])
            turned =angle_diff(turn_60_start_theta, current_theta)

            if turned >= -TURN_RIGHT:  # still need to turn right
                turtle.cmd_velocity(angular=-0.2)
                rate.sleep()
                continue
            else:
                stop()
                print("90° turn completed. Starting circle.")
                turning_right = False
                circling   = True

                # Initialise swept-angle tracker
                odom = turtle.get_odometry()
                circle_prev_phi = math.atan2(odom[1] - ball_world_y, odom[0] - ball_world_x)
                circle_swept = 0.0
                continue

        # ------------------------------------------------------------------ #
        #  Phase 3: Circle around the ball using a P-controller               #
        # ------------------------------------------------------------------ #
        elif circling and not find_garage:
           
            odom = turtle.get_odometry()
            rx, ry, rtheta = odom[0], odom[1], odom[2]

            # Radial geometry
            dx = rx - ball_world_x
            dy = ry - ball_world_y
            current_dist = math.sqrt(dx ** 2 + dy ** 2)
            dist_error   = current_dist - STOP_DISTANCE -0.05 # +ve → too far out

            phi = math.atan2(dy, dx)  # angle from ball centre → robot

            # Accumulate swept angle robustly (handles ±π wrap)
            d_phi = math.atan2(math.sin(phi - circle_prev_phi), math.cos(phi - circle_prev_phi))
            circle_swept    += math.degrees(abs(d_phi))
            circle_prev_phi  = phi

            if circle_swept >= CIRCLE_ANGLE:
                stop()
                print("Circle complete. Looking for garage.")
                circling    = False

                if ball_close  != 0 and not find_garage:
                    emergency_park()
                    print("IN GARAGE; TASK DONE")
                    turtle.play_sound(2)
                    sleep(1)
                    break
                else:
                    find_garage = True
                continue

            # Desired heading = tangent to circle (counter-clockwise)
            # Flip signs for clockwise:  math.atan2(math.cos(phi), -math.sin(phi))
            desired_heading = math.atan2(-math.cos(phi), math.sin(phi))

            heading_error = math.atan2(math.sin(desired_heading - rtheta),math.cos(desired_heading - rtheta))

            angular_ff = CIRCLE_LINEAR/STOP_DISTANCE
            angular = max(-1.0, min(1.0, angular_ff- KP_HEADING * heading_error - KD_DIST * dist_error))
            turtle.cmd_velocity(linear=CIRCLE_LINEAR, angular=angular)
            rate.sleep()
            continue

        # ------------------------------------------------------------------ #
        #  Phase 4: Find the garage (two purple pylons)                       #
        # ------------------------------------------------------------------ #
        elif find_garage and not garage_found:
            mask = get_mask(hsv, LOWER_PURPLE, UPPER_PURPLE)
            cv2.imshow('mask', mask)
            out = cv2.connectedComponentsWithStats(mask)
            indices = find_two_largest(out)

            # TODO: The same as for the ball. This search pattern is not redundant enough.
            if indices is None:
                print("Garage pylons not found... Searching ...")
                turtle.cmd_velocity(angular=-0.4)
                rate.sleep()
                continue

            # Navigate to the midpoint between the two pylons
            x1 = int(out[3][indices[0]][0])
            x2 = int(out[3][indices[1]][0])
            y1 = int(out[3][indices[0]][1])
            midpoint_x = (x1 + x2) // 2
            direction  = check_direction(midpoint_x)

            if direction != 0:
                drive(direction, 1)
                rate.sleep()
                continue

            # Distance to the closest pylon
            depth = get_depth_at(x1, y1)
            print(f"Distance from the closest pylon: {depth:.2f} m" if depth else "Distance from the pylon not available.")

            if depth is not None and depth <= STOP_DISTANCE_P:
                stop()
                garage_found = True
                print("Garage found. Task COMPLETED.")
            else:
                drive(0, 1)
                rate.sleep()

        # ------------------------------------------------------------------ #
        #  Phase 5: Parking                                                  #
        # ------------------------------------------------------------------ #
        elif garage_found:
            mask = get_mask(hsv, LOWER_PURPLE, UPPER_PURPLE)
            cv2.imshow('mask', mask)

            # Getting distances of both pylones (out and indices from last step of Phase 4)
            x1 = int(out[3][indices[0]][0])
            y1 = int(out[3][indices[0]][1])
            x2 = int(out[3][indices[1]][0])
            y2 = int(out[3][indices[1]][1])

            if x1 > x2: # Pylons were sorted base on their size; left pylon has to be left pylon
                x1, x2 = x2, x1
                y1, y2 = y2, y1

            left_dist = 100 * get_depth_at(x1, y1)
            right_dist = 100 * get_depth_at(x2, y2)

            print(f"Left p dist: {left_dist}, Right p dist: {right_dist}")
            #print(f"Lcoord: {l_coord}. Rcoord: {r_coord}")

            parking(left_dist, right_dist)

            stop()
            print("IN GARAGE; TASK DONE")
            print(f"start:", start_odo)
            print(f"door:", park_odo)
            turtle.play_sound(1)
            sleep(1)
            break


if __name__ == '__main__':
    turtle = Turtlebot(rgb=True, depth=True, pc=True)
    main()
