import numpy as np
import matplotlib.pyplot as plt
from dprelated import rotate_by_theta

# Model selection functions

def plot_angle_selection(losses, Lmax=8, rotation_angle_list=[0,5,10,15,17.5,20]):
    """
    Model selection for choosing rotation angle.
    Plot losses across different angles

    :param losses: output of rotation_dp
    :param Lmax: maximum number of layers
    :param rotation_angle_list: list of angles (in degrees) to rotate tissue by

    :return: plot of number of layers vs loss for different angles
    """

    for ind_t,angle in enumerate(rotation_angle_list):
        plt.plot(np.arange(1,Lmax+1),losses[ind_t,:], label='rotation angle: {}'.format(angle))
    plt.legend()

def plot_model_selection_nlayers(loss_list, Lmax=None):
    if Lmax is None:
        Lmax=len(loss_list)
    
    diffs=np.zeros(Lmax-1)

    for l in np.arange(2,Lmax+1):
        diffs[l-2]= (loss_list[l-2] - loss_list[l-1])

    plt.plot(np.arange(2,Lmax+1),diffs)
    plt.xlabel('Number L of layers')
    plt.title('Consecutive differences in losses')
    
def plot_model_selection_nlayers_rotation(losses, best_angle, Lmax=8, rotation_angle_list=[0,5,10,15,17.5,20]):
    """
    Model selection for choosing number L of layers.
    Plot consecutive difference of losses to identify elbow.

    :param losses: output of rotation_dp
    :param best_angle: angle (degrees) to rotate tissue slice by
    :param Lmax: maximum number of layers
    :param rotation_angle_list: list of angles (in degrees) to rotate tissue by

    :return: plot of number of consecutive difference of layers vs loss
    """
    best_angle_ind=rotation_angle_list.index(best_angle)
    losses_best_angle=losses[best_angle_ind,:]
    
    plot_model_selection_nlayers(losses_best_angle, Lmax=Lmax)

# Visualize Belayer output with given L and angle
def visualized_rotated_belayer_output(labels, coords, best_L, best_angle, rotation_angle_list=[0,5,10,15,17.5,20]):
    """
    Visualize Belayer output.

    :param labels: output of rotation_dp
    :param coords: array of size (N,2) of (x,y) coords of N spots
    :param best_L: number L of layers from model selection
    :param best_angle: rotation angle (degrees) from model selection
    :param rotation_angle_list: list of angles (in degrees) to rotate tissue by

    :return: array of size (N,) of Belayer layer labels across N spots. 
    :return: array of size (N,2) of Belayer rotated coordinates across N spots
            also plots Belayer labels across tissue slice
    """
    best_angle_ind=rotation_angle_list.index(best_angle)
    belayer_labels=labels[(best_angle_ind,best_L)]

    # plot labels

    coords_rotated=rotate_by_theta(coords,best_angle*np.pi/180)
    
    plot_cluster_output(belayer_labels, coords_rotated)
    
    # for t in np.unique(belayer_labels):
    #     pts_t=np.where(belayer_labels==t)[0]
    #     plt.scatter(coords_rotated[pts_t,0],coords_rotated[pts_t,1],s=10)
    plt.title('Belayer, L={}'.format(best_L), fontsize=16)

    return belayer_labels, coords_rotated

def plot_cluster_output(belayer_labels, coords, legend=False):
    for t in np.unique(belayer_labels):
        pts_t=np.where(belayer_labels==t)[0]
        plt.scatter(coords[pts_t,0],coords[pts_t,1],s=10, label=t)
    if legend:
        plt.legend()