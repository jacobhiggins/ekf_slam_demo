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
import pdb
import time

# <------------------------- EKF SLAM STUFF --------------------------------->
# ---> Robot Parameters
n_state = 3 # Number of state variables
robot_fov = 3 # robot field of view radius (m)

# ---> Landmark parameters
# Single landmark
# landmarks = [(7,7)] # position from lower-left corner of map (m,m)

# Scattered Landmarks
# landmarks = [(12,12),
#              (4,4),
#              (4,12),
#              (15,10),
#              (9,1)]

# Rectangular pattern
landmarks = [(4,4),
             (4,8),
             (8,8),
             (12,8),
             (16,8),
             (16,4),
             (12,4),
             (8,4)]

n_landmarks = len(landmarks)

# ---> Noise parameters
R = np.diag([0.002,0.002,0.0005]) # sigma_x, sigma_y, sigma_theta
Q = np.diag([0.003,0.005]) # sigma_r, sigma_phi

# ---> EKF Estimation Variables
mu = np.zeros((n_state+2*n_landmarks,1)) # State estimate (robot pose and landmark positions)
sigma = np.zeros((n_state+2*n_landmarks,n_state+2*n_landmarks)) # State uncertainty, covariance matrix

mu[:] = np.nan # Initialize state estimate with nan values
np.fill_diagonal(sigma,100) # Initialize state uncertainty with large variances, no correlations

# ---> Helpful matrix
Fx = np.block([[np.eye(3),np.zeros((n_state,2*n_landmarks))]]) # Used in both prediction and measurement updates

# ---> Measurement function
def sim_measurement(x,landmarks):
    '''
    This function simulates a measurement between robot and landmark
    Inputs:
     - x: robot state (3x1 numpy array)
     - landmarks: list of 2-tuples, each of (lx,ly) actual position of landmark
    Outputs:
     - zs: list of 3-tuples, each (r,phi,lidx) of range (r) and relative bearing (phi) from robot to landmark,
           and lidx is the (known) correspondence landmark index.
    '''
    rx, ry, rtheta = x[0], x[1], x[2]
    zs = [] # List of measurements
    for (lidx,landmark) in enumerate(landmarks): # Iterate over landmarks and indices
        lx,ly = landmark
        dist = np.linalg.norm(np.array([lx-rx,ly-ry])) # distance between robot and landmark
        phi = np.arctan2(ly-ry,lx-rx) - rtheta # angle between robot heading and landmark, relative to robot frame
        phi = np.arctan2(np.sin(phi),np.cos(phi)) # Keep phi bounded, -pi <= phi <= +pi
        if dist<robot_fov: # Only append if observation is within robot field of view
            zs.append((dist,phi,lidx))
    return zs

# ---> EKF SLAM steps
def prediction_update(mu,sigma,u,dt):
    '''
    This function performs the prediction step of the EKF. Using the linearized motion model, it
    updates both the state estimate mu and the state uncertainty sigma based on the model and known
    control inputs to the robot.
    Inputs:
     - mu: state estimate (robot pose and landmark positions)
     - sigma: state uncertainty (covariance matrix)
     - u: model input
     - dt: discretization time of continuous model
    Outpus:
     - mu: updated state estimate
     - sigma: updated state uncertainty
    '''
    rx,py,theta = mu[0],mu[1],mu[2]
    v,w = u[0],u[1]
    # Update state estimate mu with model
    state_model_mat = np.zeros((n_state,1)) # Initialize state update matrix from model
    state_model_mat[0] = -(v/w)*np.sin(theta)+(v/w)*np.sin(theta+w*dt) if w>0.01 else v*np.cos(theta)*dt # Update in the robot x position
    state_model_mat[1] = (v/w)*np.cos(theta)-(v/w)*np.cos(theta+w*dt) if w>0.01 else v*np.sin(theta)*dt # Update in the robot y position
    state_model_mat[2] = w*dt # Update for robot heading theta
    mu = mu + np.matmul(np.transpose(Fx),state_model_mat) # Update state estimate, simple use model with current state estimate
    # Update state uncertainty sigma
    state_jacobian = np.zeros((3,3)) # Initialize model jacobian
    state_jacobian[0,2] = (v/w)*np.cos(theta) - (v/w)*np.cos(theta+w*dt) if w>0.01 else -v*np.sin(theta)*dt # Jacobian element, how small changes in robot theta affect robot x
    state_jacobian[1,2] = (v/w)*np.sin(theta) - (v/w)*np.sin(theta+w*dt) if w>0.01 else v*np.cos(theta)*dt # Jacobian element, how small changes in robot theta affect robot y
    G = np.eye(sigma.shape[0]) + np.transpose(Fx).dot(state_jacobian).dot(Fx) # How the model transforms uncertainty
    sigma = G.dot(sigma).dot(np.transpose(G)) + np.transpose(Fx).dot(R).dot(Fx) # Combine model effects and stochastic noise
    return mu,sigma

def measurement_update(mu,sigma,zs):
    '''
    This function performs the measurement step of the EKF. Using the linearized observation model, it
    updates both the state estimate mu and the state uncertainty sigma based on range and bearing measurements
    that are made between robot and landmarks.
    Inputs:
     - mu: state estimate (robot pose and landmark positions)
     - sigma: state uncertainty (covariance matrix)
     - zs: list of 3-tuples, (dist,phi,lidx) from measurement function
    Outpus:
     - mu: updated state estimate
     - sigma: updated state uncertainty
    '''
    rx,ry,theta = mu[0,0],mu[1,0],mu[2,0] # robot 
    delta_zs = [np.zeros((2,1)) for lidx in range(n_landmarks)] # A list of how far an actual measurement is from the estimate measurement
    Ks = [np.zeros((mu.shape[0],2)) for lidx in range(n_landmarks)] # A list of matrices stored for use outside the measurement for loop
    Hs = [np.zeros((2,mu.shape[0])) for lidx in range(n_landmarks)] # A list of matrices stored for use outside the measurement for loop
    for z in zs:
        (dist,phi,lidx) = z
        mu_landmark = mu[n_state+lidx*2:n_state+lidx*2+2] # Get the estimated position of the landmark
        if np.isnan(mu_landmark[0]): # If the landmark hasn't been observed before, then initialize (lx,ly)
            mu_landmark[0] = rx + dist*np.cos(phi+theta) # lx, x position of landmark
            mu_landmark[1] = ry+ dist*np.sin(phi+theta) # ly, y position of landmark
            mu[n_state+lidx*2:n_state+lidx*2+2] = mu_landmark # Save these values to the state estimate mu
        delta  = mu_landmark - np.array([[rx],[ry]]) # Helper variable
        q = np.linalg.norm(delta)**2 # Helper variable

        dist_est = np.sqrt(q) # Distance between robot estimate and and landmark estimate, i.e., distance estimate
        phi_est = np.arctan2(delta[1,0],delta[0,0])-theta; phi_est = np.arctan2(np.sin(phi_est),np.cos(phi_est)) # Estimated angled between robot heading and landmark
        z_est_arr = np.array([[dist_est],[phi_est]]) # Estimated observation, in numpy array
        z_act_arr = np.array([[dist],[phi]]) # Actual observation in numpy array
        delta_zs[lidx] = z_act_arr-z_est_arr # Difference between actual and estimated observation

        # Helper matrices in computing the measurement update
        Fxj = np.block([[Fx],[np.zeros((2,Fx.shape[1]))]])
        Fxj[n_state:n_state+2,n_state+2*lidx:n_state+2*lidx+2] = np.eye(2)
        H = np.array([[-delta[0,0]/np.sqrt(q),-delta[1,0]/np.sqrt(q),0,delta[0,0]/np.sqrt(q),delta[1,0]/np.sqrt(q)],\
                      [delta[1,0]/q,-delta[0,0]/q,-1,-delta[1,0]/q,+delta[0,0]/q]])
        H = H.dot(Fxj)
        Hs[lidx] = H # Added to list of matrices
        Ks[lidx] = sigma.dot(np.transpose(H)).dot(np.linalg.inv(H.dot(sigma).dot(np.transpose(H)) + Q)) # Add to list of matrices
    # After storing appropriate matrices, perform measurement update of mu and sigma
    mu_offset = np.zeros(mu.shape) # Offset to be added to state estimate
    sigma_factor = np.eye(sigma.shape[0]) # Factor to multiply state uncertainty
    for lidx in range(n_landmarks):
        mu_offset += Ks[lidx].dot(delta_zs[lidx]) # Compute full mu offset
        sigma_factor -= Ks[lidx].dot(Hs[lidx]) # Compute full sigma factor
    mu = mu + mu_offset # Update state estimate
    sigma = sigma_factor.dot(sigma) # Update state uncertainty
    return mu,sigma
# <------------------------- EKF SLAM STUFF --------------------------------->


# <------------------------- PLOTTING STUFF --------------------------------->
def show_robot_estimate(mu,sigma,env):
    '''
    Visualize estimated position and uncertainty of the robot
    Inputs:
     - mu: state estimate (robot pose and landmark positions)
     - sigma: state uncertainty (covariance matrix)
     - env: Environment class (from python_ugv_sim)
    '''
    rx,ry = mu[0],mu[1]
    p_pixel = env.position2pixel((rx,ry)) # Transform robot position to pygame surface pixel coordinates
    eigenvals,angle = sigma2transform(sigma[0:2,0:2]) # Get eigenvalues and rotation angle
    sigma_pixel = env.dist2pixellen(eigenvals[0]), env.dist2pixellen(eigenvals[1]) # Convert eigenvalue units from meters to pixels
    show_uncertainty_ellipse(env,p_pixel,sigma_pixel,angle) # Show the ellipse
    
def show_landmark_estimate(mu,sigma,env):
    '''
    Visualize estimated position and uncertainty of a landmark
    '''
    for lidx in range(n_landmarks): # For each landmark location
        lx,ly,lsigma = mu[n_state+lidx*2], mu[n_state+lidx*2+1], sigma[n_state+lidx*2:n_state+lidx*2+2,n_state+lidx*2:n_state+lidx*2+2]
        if ~np.isnan(lx): # If the landmark has been observed
            p_pixel = env.position2pixel((lx,ly)) # Transform landmark location to pygame surface pixel coordinates
            eigenvals,angle = sigma2transform(lsigma) # Get eigenvalues and rotation angle of covariance of landmark
            if np.max(eigenvals)<15: # Only visualize when the maximum uncertainty is below some threshold
                sigma_pixel = max(env.dist2pixellen(eigenvals[0]),5), max(env.dist2pixellen(eigenvals[1]),5) # Convert eigenvalue units from meters to pixels
                show_uncertainty_ellipse(env,p_pixel,sigma_pixel,angle) # Show the ellipse

def show_landmark_location(landmarks,env):
    '''
    Visualize actual landmark location
    '''
    for landmark in landmarks:
        lx_pixel, ly_pixel = env.position2pixel(landmark)
        r_pixel = env.dist2pixellen(0.2) # Radius of the circle for the ground truth locations of the landmarks
        pygame.gfxdraw.filled_circle(env.get_pygame_surface(),lx_pixel,ly_pixel,r_pixel,(0,255,255)) # Blit the circle onto the surface

def show_measurements(x,zs,env):
    '''
    Visualize measurements the occur between the robot and landmarks
    '''
    rx,ry = x[0], x[1]
    rx_pix, ry_pix = env.position2pixel((rx,ry)) # Convert robot position units from meters to pixels
    for z in zs: # For each measurement
        dist,theta,lidx = z # Unpack measurement tuple
        lx,ly = x[0]+dist*np.cos(theta+x[2]),x[1]+dist*np.sin(theta+x[2]) # Set the observed landmark location (lx,ly)
        lx_pix,ly_pix = env.position2pixel((lx,ly)) # Convert observed landmark location units from meters to pixels
        pygame.gfxdraw.line(env.get_pygame_surface(),rx_pix,ry_pix,lx_pix,ly_pix,(155,155,155)) # Draw a line between robot and observed landmark

def sigma2transform(sigma):
    '''
    Finds the transform for a covariance matrix, to be used for visualizing the uncertainty ellipse
    '''
    [eigenvals,eigenvecs] = np.linalg.eig(sigma) # Finding eigenvalues and eigenvectors of the covariance matrix
    angle = 180.*np.arctan2(eigenvecs[1][0],eigenvecs[0][0])/np.pi # Find the angle of rotation for the first eigenvalue
    return eigenvals, angle

def show_uncertainty_ellipse(env,center,width,angle):
    '''
    Visualize an uncertainty ellipse
    Adapted from: https://stackoverflow.com/questions/65767785/how-to-draw-a-rotated-ellipse-using-pygame
    '''
    target_rect = pygame.Rect(center[0]-int(width[0]/2),center[1]-int(width[1]/2),width[0],width[1])
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.ellipse(shape_surf, env.red, (0, 0, *target_rect.size), 2)
    rotated_surf = pygame.transform.rotate(shape_surf, angle)
    env.get_pygame_surface().blit(rotated_surf, rotated_surf.get_rect(center = target_rect.center))

# <------------------------- PLOTTING STUFF --------------------------------->

if __name__ == '__main__':

    # Initialize pygame
    pygame.init()
    
    # Initialize robot and discretization time step
    x_init = np.array([1,1,np.pi/2]) # px, py, theta
    robot = vehicles.DifferentialDrive(x_init)
    dt = 0.01

    # Initialize and display environment
    env = environment.Environment(map_image_path="./python_ugv_sim/maps/map_blank.png")

    # Initialize robot state estimate and sigma
    mu[0:3] = np.expand_dims(x_init,axis=1)
    sigma[0:3,0:3] = 0.1*np.eye(3)
    sigma[2,2] = 0 
    running = True
    u = np.array([0.,0.]) # Controls: u[0] = forward velocity, u[1] = angular velocity
    while running:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running = False
            u = robot.update_u(u,event) if event.type==pygame.KEYUP or event.type==pygame.KEYDOWN else u # Update controls based on key states
        # Move the robot
        robot.move_step(u,dt) # Integrate EOMs forward, i.e., move robot
        # Get measurements
        zs = sim_measurement(robot.get_pose(),landmarks) # Simulate measurements between robot and landmarks
        # EKF Slam Logic
        mu, sigma = prediction_update(mu,sigma,u,dt) # Perform EKF prediction update
        mu, sigma = measurement_update(mu,sigma,zs) # Perform EKF measurement update
        # Plotting
        env.show_map() # Re-blit map
        # Show measurements
        show_measurements(robot.get_pose(),zs,env)
        # Show actual locations of robot and landmarks
        env.show_robot(robot) # Re-blit robot
        show_landmark_location(landmarks,env)
        # Show estimates of robot and landmarks (estimate and uncertainty)
        show_robot_estimate(mu,sigma,env)
        show_landmark_estimate(mu,sigma,env)

        pygame.display.update() # Update display