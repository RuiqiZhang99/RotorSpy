import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline

from matplotlib import animation

from pyplot3d.uav import Uav
from pyplot3d.utils import ypr_to_R
import pandas as pd



def animate_quadcopter_history(times, x, R, arm_length=0.3, tilt_angle=0.0, gif_path="data/quad_sim_animation.gif"):
    plt.style.use('ggplot')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    space_lim = (0, 2)
    uav_plot = Uav(ax, arm_length,tilt_angle=tilt_angle) # TODO: change hardcode
    
    def update_plot(i):
        
        # ax.cla()
        uav_plot.draw_at(x[:, i], R[:, :, i])
        
        # These limits must be set manually since we use
        # a different axis frame configuration than the
        # one matplotlib uses.
        
        ax.set_xlim(space_lim)
        ax.set_ylim(space_lim)
        ax.set_zlim((0, space_lim[1]))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])
        # ax.invert_zaxis()
        ax.set_title("Quadcopter Animation (Time: {0:.3f} s)".format(times[i]))
    
    # animate @ 1/desired_interval hz, with data from every step_size * dt
    animate_interval = 50
    step_size = 25
    steps = len(times)
    ind = [i * step_size for i in range(steps // step_size)]
    ani = animation.FuncAnimation(fig, update_plot, frames=ind, interval=animate_interval);
    # Define the filename for the GIF
    output_gif_path = gif_path
    # Save the animation as a GIF
    ani.save(output_gif_path, writer='pillow')
    # Display the animation
    plt.show()
    
def animate_multidrones_history(times, x_list, R_list, armLength_list, tilt_angle=0.0, gif_path="data/multi_drones_animation.gif"):
    plt.style.use('ggplot')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    space_lim = (0, 2)
    
    def update_plot(i):
        ax.clear()
        # ax.cla()
        assert len(x_list) == len(R_list)
        assert len(x_list) == len(armLength_list)
        for uav_id in range(len(x_list)):
            x, R, armLength = x_list[uav_id], R_list[uav_id], armLength_list[uav_id]
            uav_plot = Uav(ax, armLength, tilt_angle=tilt_angle)
            uav_plot.draw_at(x[:, i], R[:, :, i], auto_clear=False)
        
        # These limits must be set manually since we use
        # a different axis frame configuration than the
        # one matplotlib uses.
        
        ax.set_xlim(space_lim)
        ax.set_ylim(space_lim)
        ax.set_zlim((0, space_lim[1]))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])
        # ax.invert_zaxis()
        ax.set_title("Quadcopter Animation (Time: {0:.3f} s)".format(times[i]))
    
    # animate @ 1/desired_interval hz, with data from every step_size * dt
    animate_interval = 50
    step_size = 25
    steps = len(times)
    ind = [i * step_size for i in range(steps // step_size)]
    ani = animation.FuncAnimation(fig, update_plot, frames=ind, interval=animate_interval);
    # Define the filename for the GIF
    output_gif_path = gif_path
    # Save the animation as a GIF
    ani.save(output_gif_path, writer='pillow')
    # Display the animation
    plt.show()

# # Create some fake simulation data
# steps = 500
# t_end = 1
# times = np.linspace(0, t_end, steps)
# x = np.zeros((3, steps))
# x[0, :] = np.arange(0, t_end, t_end / steps)
# x[1, :] = np.arange(0, t_end, t_end / steps) * 2

# R = np.zeros((3, 3, steps))
# for i in range(steps):
#     ypr = np.array([0, 0.5*i, 0.0])
#     R[:, :, i] = ypr_to_R(ypr, degrees=True)


# # TODO: change constants

# animate_quadcopter_history(times, x, R)