'''
EKF SLAM demo
Logic:
    - Prediction update
        - From control inputs, how do we change our state estimate?
        - Moving only changes the state estimate of the robot state, NOT landmark location
        - Moving affects uncertainty of the state
    - Observation update
        - From what we observe, how do we change our state estimation?
        - We reconcile prediction uncertainty and observation uncertainty into a single estimate
          that is more certain than before
'''
import numpy as np
import pygame
import pygame.gfxdraw
from python_ugv_sim.utils import environment, vehicles

# <------------------------- EKF SLAM STUFF --------------------------------->
# Sim Parameters
n_state = 3
n_landmarks = 5
robot_fov = 5

landmarks = [(12,12),
             (5,5),
             (4,12),
             (15,10),
             (9,1)]

# Noise parameters

# Variables
mu = np.zeros((n_state+2*n_landmarks,1)) # estimation of state and landmarks
sigma = np.empty((n_state+2*n_landmarks,n_state+2*n_landmarks)) # standard deviation of state and landmark uncertainty

# Helpful matrices
global Fx
Fx = np.block([[np.eye(3),np.zeros((3,2*n_landmarks))]])

# Measurement function
def sim_measurement(x,landmarks):
    zs = []
    for landmark in landmarks:
        lx,ly = landmark
        dist = np.linalg.norm(x[0:2]-np.array([lx,ly]))
        theta = np.arctan2(ly-x[1],lx-x[0]) - x[2]
        theta = np.arctan2(np.sin(theta),np.cos(theta))
        if dist<robot_fov:
            zs.append((dist,theta))
    return zs

# EKF SLAM steps
def prediction_update(mu,sigma,u,dt):
    global Fx
    px,py,theta = mu[0],mu[1],mu[2]
    v,w = u[0],u[1]
    state_model_mat = np.zeros((3,1))
    state_model_mat[0] = -(v/w)*np.sin(theta)+(v/w)*np.sin(theta+w*dt) if w>0.01 else v*np.cos(theta)*dt
    state_model_mat[1] = (v/w)*np.cos(theta)-(v/w)*np.cos(theta+w*dt) if w>0.01 else v*np.sin(theta)*dt
    state_model_mat[2] = w*dt
    # Update mu and sigma
    mu = mu + np.matmul(np.transpose(Fx),state_model_mat)
    return mu,sigma
def measurement_update(mu,sigma,z):

    return mu,sigma
# <------------------------- EKF SLAM STUFF --------------------------------->


# Plotting functions
def show_robot_estimate(mu,sigma,env):
    px,py,sigmax,sigmay = mu[0],mu[1],1,1
    px_pixel,py_pixel = env.position2pixel((px,py))
    sigmax_pixel,sigmay_pixel = env.dist2pixellen(sigmax), env.dist2pixellen(sigmay)
    pygame.gfxdraw.aaellipse(env.get_pygame_surface(),px_pixel,py_pixel,sigmax_pixel,sigmay_pixel,(255,0,0))
    pass
def show_landmark_location(landmarks,env):
    for landmark in landmarks:
        lx_pixel, ly_pixel = env.position2pixel(landmark)
        r_pixel = env.dist2pixellen(0.2)
        pygame.gfxdraw.filled_circle(env.get_pygame_surface(),lx_pixel,ly_pixel,r_pixel,(0,255,255))
def show_measurements(x,zs,env):
    rx,ry = x[0], x[1]
    rx_pix, ry_pix = env.position2pixel((rx,ry))
    for z in zs:
        dist,theta = z
        lx,ly = x[0]+dist*np.cos(theta+x[2]),x[1]+dist*np.sin(theta+x[2])
        lx_pix,ly_pix = env.position2pixel((lx,ly))
        pygame.gfxdraw.line(env.get_pygame_surface(),rx_pix,ry_pix,lx_pix,ly_pix,(155,155,155))

if __name__ == '__main__':

    # Initialize pygame
    pygame.init()
    
    # Initialize robot and time step
    x_init = np.array([1,1,np.pi/2]) # px, py, theta
    robot = vehicles.DifferentialDrive(x_init)
    dt = 0.01

    # Initialize and display environment
    env = environment.Environment(map_image_path="./python_ugv_sim/maps/map_blank.png")

    # Initialize robot state estimate
    mu[0:3] = np.expand_dims(x_init,axis=1)
    
    running = True
    u = np.array([0.,0.]) # Controls: u[0] = forward velocity, u[1] = angular velocity
    while running:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running = False
            u = robot.update_u(u,event) if event.type==pygame.KEYUP or event.type==pygame.KEYDOWN else u # Update controls based on key states
        # Movement
        robot.move_step(u,dt) # Integrate EOMs forward, i.e., move robot
        # Get measurements
        zs = sim_measurement(robot.get_pose(),landmarks)
        # EKF Slam Logic
        mu, sigma = prediction_update(mu,sigma,u,dt)
        # Plotting
        env.show_map() # Re-blit map
        env.show_robot(robot) # Re-blit robot
        show_measurements(robot.get_pose(),zs,env)
        show_robot_estimate(mu,sigma,env)
        show_landmark_location(landmarks,env)
        pygame.display.update() # Update display