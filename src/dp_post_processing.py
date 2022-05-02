import numpy as np
import matplotlib.pyplot as plt
from utils import rotate_by_theta

# Model selection

def plot_angle_selection(losses, Lmax=8, rotation_angle_list=[0,5,10,15,17.5,20]):
    for ind_t,angle in enumerate(rotation_angle_list):
        plt.plot(np.arange(1,Lmax+1),losses[ind_t,:], label='rotation angle: {}'.format(angle))
    plt.legend()

def plot_consecutive_diffs(losses, best_angle, Lmax=8, rotation_angle_list=[0,5,10,15,17.5,20]):
    best_angle_ind=rotation_angle_list.index(best_angle)
    losses_best_angle=losses[best_angle_ind,:]

    diffs=np.zeros(Lmax-1)

    for l in np.arange(2,Lmax+1):
        diffs[l-2]= (losses_best_angle[l-2] - losses_best_angle[l-1])

    plt.plot(np.arange(2,Lmax+1),diffs)
    plt.xlabel('Number L of layers')
    plt.title('Consecutive differences in losses')

# Visualize Belayer output with given L and angle
def belayer_output(labels, coords, best_L, best_angle, rotation_angle_list=[0,5,10,15,17.5,20]):
    best_angle_ind=rotation_angle_list.index(best_angle)
    belayer_labels=labels[(best_angle_ind,best_L)]

    # plot labels
    print(rotate_by_theta)
    coords_rotated=rotate_by_theta(coords,best_angle*np.pi/180)
    for t in np.unique(belayer_labels):
        pts_t=np.where(belayer_labels==t)[0]
        plt.scatter(coords_rotated[pts_t,0],coords_rotated[pts_t,1],s=10)
    plt.title('Belayer, L={}'.format(best_L), fontsize=16)

    return belayer_labels