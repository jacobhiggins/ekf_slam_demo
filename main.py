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
from python_ugv_sim.utils import environment, vehicles

# <------------------------- EKF SLAM STUFF --------------------------------->
# Sim Parameters
n_state = 3
n_landmarks = 5

# Noise parameters

# Variables
mu = np.zeros((n_state+2*n_landmarks,1)) # estimation of state and landmarks
sigma = np.empty((n_state+2*n_landmarks,n_state+n_landmarks)) # standard deviation of state and landmark uncertainty

# Helpful matrices
global Fx
Fx = np.block([[np.eye(3),np.zeros((3,2*n_landmarks))]])

# EKF SLAM steps
def prediction_update(mu,sigma,u,dt):
    global Fx
    px,py,theta = mu[0],mu[1],mu[2]
    v,w = u[0],u[1]
    state_model_mat = np.zeros((3,1))
    state_model_mat[0] = -(v/w)*np.sin(theta)+(v/w)*np.sin(theta+w*dt) if w>0.1 else v*np.cos(theta)
    state_model_mat[1] = (v/w)*np.cos(theta)-(v/w)*np.cos(theta+w*dt) if w>0.1 else v*np.sin(theta)

    # Update mu and sigma
    mu = mu + np.matmul(np.transpose(Fx),state_model_mat)
    return mu,sigma
# <------------------------- EKF SLAM STUFF --------------------------------->


# Plotting functions
def plot_robot_estimate(mu,sigma,pygame_surface):
    # Get rectangle corners
    pass

if __name__ == '__main__':

    # Initialize pygame
    pygame.init()
    
    # Initialize robot and time step
    x_init = np.array([1,1,np.pi/2]) # px, py, theta
    robot = vehicles.DifferentialDrive(x_init)
    dt = 0.01

    # Initialize and display environment
    env = environment.Environment(map_image_path="./python_ugv_sim/maps/map_blank.png")
    
    running = True
    u = np.array([0.,0.]) # Controls: u[0] = forward velocity, u[1] = angular velocity
    while running:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running = False
            u = robot.update_u(u,event) if event.type==pygame.KEYUP or event.type==pygame.KEYDOWN else u # Update controls based on key states
        robot.move_step(u,dt) # Integrate EOMs forward, i.e., move robot
        env.show_map() # Re-blit map
        env.show_robot(robot) # Re-blit robot
        pygame.display.update() # Update display