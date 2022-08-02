import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
import cv2
import time
from findiff import FinDiff


# Settings:
filename = "1_1_60um_1mm_lattice_evolution_4x_GreenRed_SpectraIII_Cyan3_Yellow100_bright_low_res(1d4).m4v"
dt = 0.1666733  # sec
scaling = 1.625  # um/pixel
start_trajectories = 300-25  # time in sec when the trajectories begin (14)
end_trajectories = 400  # time in sec when the trajectories end (400)
of_winsize = 15  # (15)
powers = [1]   # [1, 2, 3, 4]
accuracy = 2  # 2, 4, 6, 8

make_animation = False  # for time-dependent EP
time_window_duration = 25  # duration of window for calculating time-dependent EP, in sec
fps = 20

output_filename_suffix = ""  # f"(tw{time_window_duration}s)"


d_dx = FinDiff(1, 1, 1, acc=accuracy)
d_dy = FinDiff(0, 1, 1, acc=accuracy)

if make_animation:
    vorticity_sum_sequence = []
    vorticity_sum_abs_sequence = []
    fig, ax = plt.subplots()


def cal_vorticity(flow):
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    vx, vy = fx/dt, fy/dt   # in pixels/sec
#    dvx_dy = np.gradient(vx, axis=0)
#    dvy_dx = np.gradient(vy, axis=1)
    dvx_dy = d_dy(vx)
    dvy_dx = d_dx(vy)
    vorticity = dvy_dx - dvx_dy
    return vorticity


def normalize_to_max_one(arr):
    return arr/np.amax(np.absolute(arr))


def init():
    pass


def animate(i):
    ax.cla()
    fig.suptitle(f"Time: {start_trajectories + time_window_duration/2 + i*dt:.2f} s", fontsize=12)
    heatmap = ax.imshow(vorticity_sum_abs_sequence[i], cmap='inferno', interpolation='none', vmin=0, vmax=1)
    if i == 0:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(heatmap, cax=cax)
        cbar.ax.tick_params(labelsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    print(f"Making animation frame {i+1} of {ani_length}")


# Main Program:
start_time = time.perf_counter()

cap = cv2.VideoCapture(filename)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
i = 1

vorticity_sequence = []
while True:
    ret, img = cap.read()
    if not ret:
        break
    i += 1
    print(f"Processing frame {i} of {length}")
    if start_trajectories < (i-1)*dt < end_trajectories:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, of_winsize, 3, 5, 1.2, 0)
        prevgray = gray
        vorticity_sequence.append(cal_vorticity(flow))

if make_animation:

    time_window_size = int(time_window_duration/dt)+1   # in no. of frames
    length = len(vorticity_sequence)
    hi = time_window_size
    lo = 0

    while hi <= length:
        print(f"Summing frame {lo+1} to {hi} of {length}")
        vorticity_sum = np.zeros_like(vorticity_sequence[0])
        for vorticity in vorticity_sequence[lo:hi]:
            vorticity_sum += vorticity
        vorticity_sum_abs = normalize_to_max_one(np.absolute(vorticity_sum[1:-1, 1:-1]))
        vorticity_sum_abs_sequence.append(vorticity_sum_abs)
        lo += 1
        hi += 1

    ani_length = len(vorticity_sum_abs_sequence)
    ani = animation.FuncAnimation(fig, animate, frames=ani_length, interval=10, repeat=False, init_func=init)
    ani.save(filename[:-4]+"_vorticity_sum_normalized_abs_vs_t"+output_filename_suffix+filename[-4:], writer='ffmpeg', fps=fps, dpi=300)

else:

    vorticity_sum = np.zeros_like(vorticity_sequence[0])
    for vorticity in vorticity_sequence:
        vorticity_sum += vorticity
    vorticity_sum = vorticity_sum[1:-1, 1:-1]

    plot_queue = []
    for power in powers:
        vorticity_sum_power_normalized = normalize_to_max_one(np.power(vorticity_sum, power))
        is_odd = power % 2
        plot_queue.append([vorticity_sum_power_normalized, f"_power{power}_normalized", is_odd])
        if is_odd:
            plot_queue.append([np.absolute(vorticity_sum_power_normalized), f"_power{power}_normalized_abs", 0])

    for plot in plot_queue:
        fig, ax = plt.subplots()
        if plot[2] == 1:
            heatmap = ax.imshow(plot[0], cmap='seismic', interpolation='none', vmin=-1, vmax=1)
        else:
            heatmap = ax.imshow(plot[0], cmap='inferno', interpolation='none', vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        if plot[2] == 1:
            cbar = fig.colorbar(heatmap, cax=cax, ticks=np.arange(-1, 1+0.2, 0.2))
        else:
            cbar = fig.colorbar(heatmap, cax=cax, ticks=np.arange(0, 1+0.2, 0.2))
        cbar.ax.tick_params(labelsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        plt.savefig(filename[:-4] + "_vorticity_sum" + plot[1] + f"_acc{accuracy}" + output_filename_suffix + ".png", dpi=300)

end_time = time.perf_counter()
print(f"\nTotal run time: {end_time-start_time: .2f} s")
print("\a")
