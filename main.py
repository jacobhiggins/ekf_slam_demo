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

R = np.diag([0.001,0.001,0.0]) # sigma_x, sigma_y, sigma_theta
Q = np.diag([0.1,0.1]) # sigma_r, sigma_phi

# Noise parameters

# Variables
mu = np.empty((n_state+2*n_landmarks,1)) # estimation of state and landmarks
sigma = np.zeros((n_state+2*n_landmarks,n_state+2*n_landmarks)) # standard deviation of state and landmark uncertainty

mu[:] = np.nan
np.fill_diagonal(sigma,1000)

# Helpful matrices
Fx = np.block([[np.eye(3),np.zeros((3,2*n_landmarks))]])

# Measurement function
def sim_measurement(x,landmarks):
    zs = []
    for (lidx,landmark) in enumerate(landmarks):
        lx,ly = landmark
        dist = np.linalg.norm(x[0:2]-np.array([lx,ly]))
        phi = np.arctan2(ly-x[1],lx-x[0]) - x[2]
        phi = np.arctan2(np.sin(phi),np.cos(phi))
        if dist<robot_fov:
            zs.append((dist,phi,lidx))
    return zs

# EKF SLAM steps
def prediction_update(mu,sigma,u,dt):
    
    px,py,theta = mu[0],mu[1],mu[2]
    v,w = u[0],u[1]
    # Update mu
    state_model_mat = np.zeros((3,1))
    state_model_mat[0] = -(v/w)*np.sin(theta)+(v/w)*np.sin(theta+w*dt) if w>0.01 else v*np.cos(theta)*dt
    state_model_mat[1] = (v/w)*np.cos(theta)-(v/w)*np.cos(theta+w*dt) if w>0.01 else v*np.sin(theta)*dt
    state_model_mat[2] = w*dt
    mu = mu + np.matmul(np.transpose(Fx),state_model_mat)
    # Update sigma
    state_jacobian = np.zeros((3,3))
    state_jacobian[0,2] = (v/w)*np.cos(theta) - (v/w)*np.cos(theta+w*dt) if w>0.1 else -v*np.sin(theta)*dt
    state_jacobian[1,2] = (v/w)*np.sin(theta) - (v/w)*np.sin(theta+w*dt) if w>0.1 else v*np.cos(theta)*dt
    G = np.eye(sigma.shape[0]) + np.transpose(Fx).dot(state_jacobian).dot(Fx)
    sigma = G.dot(sigma).dot(np.transpose(G)) + np.transpose(Fx).dot(R).dot(Fx)
    # sigma_old = sigma # Trick for the multiplication to work out
    # sigma = G.dot(np.nan_to_num(sigma)).dot(np.transpose(G)) + np.transpose(Fx).dot(R).dot(Fx)
    # sigma[np.isnan(sigma_old)] = np.nan # Anything that was nan before should be nan now
    return mu,sigma
    
def measurement_update(mu,sigma,zs):
    px,py,theta = mu[0,0],mu[1,0],mu[2,0]
    delta_zs = [np.zeros((2,1)) for lidx in range(n_landmarks)]
    Ks = [np.zeros((mu.shape[0],2)) for lidx in range(n_landmarks)]
    Hs = [np.zeros((2,mu.shape[0])) for lidx in range(n_landmarks)]
    for z in zs:
        (dist,phi,lidx) = z
        mu_landmark = mu[3+lidx*2:3+lidx*2+2]
        if np.isnan(mu_landmark[0]):
            mu_landmark[0] = px + dist*np.cos(phi+theta)
            mu_landmark[1] = py + dist*np.sin(phi+theta)
            mu[3+lidx*2:3+lidx*2+2] = mu_landmark
        delta  = mu_landmark - np.array([[px],[py]])
        q = np.linalg.norm(delta)**2
        # Estimated and actual observation for this landmark
        z_est_arr = np.array([[np.sqrt(q)],[np.arctan2(delta[1,0],delta[0,0])-theta]])
        z_act_arr = np.array([[dist],[phi]])
        delta_zs[lidx] = z_act_arr-z_est_arr
        # Get matrices
        Fxj = np.block([[Fx],[np.zeros((2,Fx.shape[1]))]])
        Fxj[3:5,3+2*lidx:3+2*lidx+2] = np.eye(2)
        H = np.array([[-delta[0,0]/np.sqrt(q),-delta[1,0]/np.sqrt(1),0,delta[0,0]/np.sqrt(q),delta[1,0]/np.sqrt(q)],\
                      [delta[1,0]/q,-delta[0,0]/q,-1,-delta[1,0]/q,-delta[0,0]/q]])
        H = H.dot(Fxj)
        Hs[lidx] = H
        Ks[lidx] = sigma.dot(np.transpose(H))*np.linalg.inv(H.dot(sigma).dot(np.transpose(H)) + Q)
    mu_alteration = np.zeros(mu.shape)
    sigma_alteration = np.eye(sigma.shape[0])
    for lidx in range(n_landmarks):
        mu_alteration += Ks[lidx].dot(delta_zs[lidx])
        # pdb.set_trace()
        sigma_alteration -= Ks[lidx].dot(Hs[lidx])
    mu = mu + mu_alteration
    sigma = sigma_alteration.dot(sigma)
    return mu,sigma
# <------------------------- EKF SLAM STUFF --------------------------------->


# Plotting functions
def show_robot_estimate(mu,sigma,env):
    px,py,sigmax,sigmay = mu[0],mu[1],sigma[0,0],sigma[1,1]

    p_pixel = env.position2pixel((px,py))
    eigenvals,angle = sigma2transform(sigma[0:2,0:2])
    sigma_pixel = env.dist2pixellen(eigenvals[0]), env.dist2pixellen(eigenvals[1])
    show_uncertainty_ellipse(env,p_pixel,sigma_pixel,angle)

    # px_pixel,py_pixel = env.position2pixel((px,py))
    # sigmax_pixel,sigmay_pixel = env.dist2pixellen(sigmax), env.dist2pixellen(sigmay)
    # pygame.gfxdraw.aaellipse(env.get_pygame_surface(),px_pixel,py_pixel,sigmax_pixel,sigmay_pixel,(255,0,0))
    
def show_landmark_location(landmarks,env):
    for landmark in landmarks:
        lx_pixel, ly_pixel = env.position2pixel(landmark)
        r_pixel = env.dist2pixellen(0.2)
        pygame.gfxdraw.filled_circle(env.get_pygame_surface(),lx_pixel,ly_pixel,r_pixel,(0,255,255))
def show_measurements(x,zs,env):
    rx,ry = x[0], x[1]
    rx_pix, ry_pix = env.position2pixel((rx,ry))
    for z in zs:
        dist,theta,lidx = z
        lx,ly = x[0]+dist*np.cos(theta+x[2]),x[1]+dist*np.sin(theta+x[2])
        lx_pix,ly_pix = env.position2pixel((lx,ly))
        pygame.gfxdraw.line(env.get_pygame_surface(),rx_pix,ry_pix,lx_pix,ly_pix,(155,155,155))
def sigma2transform(sigma):
    [eigenvals,eigenvecs] = np.linalg.eig(sigma)
    angle = 180.*np.arctan2(eigenvecs[1][0],eigenvecs[0][0])/np.pi
    return eigenvals, angle
def show_uncertainty_ellipse(env,center,width,angle):
    target_rect = pygame.Rect(center[0]-int(width[0]/2),center[1]-int(width[1]/2),width[0],width[1])
    # target_rect.center = center
    # target_rect.size = width
    # pdb.set_trace()
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.ellipse(shape_surf, env.red, (0, 0, *target_rect.size), 2)
    rotated_surf = pygame.transform.rotate(shape_surf, angle)
    env.get_pygame_surface().blit(rotated_surf, rotated_surf.get_rect(center = target_rect.center))

def draw_ellipse_angle(surface, color, rect, angle, width=0):
    target_rect = pygame.Rect(rect)
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.ellipse(shape_surf, color, (0, 0, *target_rect.size), width)
    rotated_surf = pygame.transform.rotate(shape_surf, angle)
    surface.blit(rotated_surf, rotated_surf.get_rect(center = target_rect.center))

if __name__ == '__main__':

    # Initialize pygame
    pygame.init()
    
    # Initialize robot and time step
    x_init = np.array([1,1,np.pi/2]) # px, py, theta
    robot = vehicles.DifferentialDrive(x_init)
    dt = 0.01

    # Initialize and display environment
    env = environment.Environment(map_image_path="./python_ugv_sim/maps/map_blank.png")

    # Initialize robot state estimate and sigma
    mu[0:3] = np.expand_dims(x_init,axis=1)
    sigma[0:3,0:3] = 0.1*np.eye(3) 
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
        mu, sigma = measurement_update(mu,sigma,zs)
        # Plotting
        env.show_map() # Re-blit map
        # Show measurements
        show_measurements(robot.get_pose(),zs,env)
        # Show actual locations of robot and landmarks
        env.show_robot(robot) # Re-blit robot
        show_landmark_location(landmarks,env)
        # Show estimates of robot and landmarks (estimate and uncertainty)
        show_robot_estimate(mu,sigma,env)

        pygame.display.update() # Update display