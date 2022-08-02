import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
import cv2
import time


# Settings:
filename = "1_1_60um_1mm_lattice_evolution_4x_GreenRed_SpectraIII_Cyan3_Yellow100_low_res(1d4).m4v"
dt = 0.1666733  # sec
scaling = 1.625  # um/pixel
start_trajectories = 14  # time in sec when the trajectories begin (14)
end_trajectories = 400  # time in sec when the trajectories end (400)
of_winsize = 30  # (15)
occupancy_threshold_speeds = [0.1]  # ([0, 0.001, 0.01, 0.1, 1, 10])
apply_filters = 0   # 1 or 0

make_animation = True  # for time-dependent EP
time_window_duration = 25  # duration of window for calculating time-dependent EP, in sec
fps = 20

output_filename_suffix = f"(tw{time_window_duration}s)"


if make_animation:
    ani_sequence = []
    fig, ax = plt.subplots()


def cal_vel(flow):
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    v = v*scaling/dt
    vel = np.stack((ang, v), axis=-1)
    return vel


def cal_cross_parsing_length(Y, Z):

    y = ''
    z = ''
    for elem in Y:
        y = y + chr(elem+19968)
    for elem in Z:
        z = z + chr(elem+19968)
    y_len = len(y)

    C = 0
    ptr = 0
    while ptr < y_len:
        l = 1
        while y[ptr:ptr+l] in z and ptr+l <= y_len:
            l += 1
        C += 1
        if l == 1:
            ptr += 1
        else:
            ptr = ptr+l-1

    return C


def map_to_state(vel, occupancy_threshold_speed):

    orientations = []
    # ignore orientations
#    orientations.append((vel[:, :, 0] >= 0))
    # or4: left, right, up, down
#    orientations.append((vel[:, :, 0] >= 0) & (vel[:, :, 0] <= np.pi/4) & (vel[:, :, 0] > np.pi*7/4) & (vel[:, :, 0] <= np.pi*2))  # left
#    orientations.append((vel[:, :, 0] > np.pi/4) & (vel[:, :, 0] <= np.pi*3/4))    # down
#    orientations.append((vel[:, :, 0] > np.pi*3/4) & (vel[:, :, 0] <= np.pi*5/4))  # right
#    orientations.append((vel[:, :, 0] > np.pi*5/4) & (vel[:, :, 0] <= np.pi*7/4))  # up
    # or4d: diagonal
#    orientations.append((vel[:, :, 0] >= 0) & (vel[:, :, 0] <= np.pi/2))  # quadrant 3
#    orientations.append((vel[:, :, 0] > np.pi/2) & (vel[:, :, 0] <= np.pi))  # quadrant 4
#    orientations.append((vel[:, :, 0] > np.pi) & (vel[:, :, 0] <= np.pi*3/2))   # quadrant 1
#    orientations.append((vel[:, :, 0] > np.pi*3/2) & (vel[:, :, 0] <= np.pi*2))  # quadrant 2
    # or8: 8 orientaions (left, right, up, down + diagonal)
    orientations.append((vel[:, :, 0] >= 0) & (vel[:, :, 0] <= np.pi/8) & (vel[:, :, 0] > np.pi*15/8) & (vel[:, :, 0] <= np.pi*2))  # left
    orientations.append((vel[:, :, 0] > np.pi/8) & (vel[:, :, 0] <= np.pi*3/8))
    orientations.append((vel[:, :, 0] > np.pi*3/8) & (vel[:, :, 0] <= np.pi*5/8))  # down
    orientations.append((vel[:, :, 0] > np.pi*5/8) & (vel[:, :, 0] <= np.pi*7/8))
    orientations.append((vel[:, :, 0] > np.pi*7/8) & (vel[:, :, 0] <= np.pi*9/8))  # right
    orientations.append((vel[:, :, 0] > np.pi*9/8) & (vel[:, :, 0] <= np.pi*11/8))
    orientations.append((vel[:, :, 0] > np.pi*11/8) & (vel[:, :, 0] <= np.pi*13/8))    # up
    orientations.append((vel[:, :, 0] > np.pi*13/8) & (vel[:, :, 0] <= np.pi*15/8))
    # or8d: 8 diagonal
#    orientations.append((vel[:, :, 0] >= 0) & (vel[:, :, 0] <= np.pi/4))  # quadrant 3
#    orientations.append((vel[:, :, 0] > np.pi/4) & (vel[:, :, 0] <= np.pi/2))  # quadrant 3
#    orientations.append((vel[:, :, 0] > np.pi/2) & (vel[:, :, 0] <= np.pi*3/4))  # quadrant 4
#    orientations.append((vel[:, :, 0] > np.pi*3/4) & (vel[:, :, 0] <= np.pi))  # quadrant 4
#    orientations.append((vel[:, :, 0] > np.pi) & (vel[:, :, 0] <= np.pi*5/4))   # quadrant 1
#    orientations.append((vel[:, :, 0] > np.pi*5/4) & (vel[:, :, 0] <= np.pi*3/2))   # quadrant 1
#    orientations.append((vel[:, :, 0] > np.pi*3/2) & (vel[:, :, 0] <= np.pi*7/4))  # quadrant 2
#    orientations.append((vel[:, :, 0] > np.pi*7/4) & (vel[:, :, 0] <= np.pi*2))  # quadrant 2

    speeds = []
    # occupancy only
    speeds.append((vel[:, :, 1] > occupancy_threshold_speed))
    # occupancy + speed
#    speeds.append((vel[:, :, 1] > occupancy_threshold_speed) & (vel[:, :, 1] <= 50))
#    speeds.append((vel[:, :, 1] > 50) & (vel[:, :, 1] <= 100))
#    speeds.append((vel[:, :, 1] > 100) & (vel[:, :, 1] <= 150))
#    speeds.append((vel[:, :, 1] > 150) & (vel[:, :, 1] <= 200))
#    speeds.append((vel[:, :, 1] > 200) & (vel[:, :, 1] <= 250))
#    speeds.append((vel[:, :, 1] > 250) & (vel[:, :, 1] <= 300))
#    speeds.append((vel[:, :, 1] > 300) & (vel[:, :, 1] <= 350))
#    speeds.append((vel[:, :, 1] > 350) & (vel[:, :, 1] <= 400))
#    speeds.append((vel[:, :, 1] > 400))

    state = np.zeros_like(vel[:, :, 0], dtype=int)
    b = 1
    for speed in speeds:
        for orientation in orientations:
            state += b*(speed & orientation).astype(int)
            b += 1

    state_2x2 = np.zeros_like(state, dtype=int)
    h, w = state_2x2.shape
    for i, j in np.ndindex(h-1, w-1):
        state_2x2[i, j] = state[i, j] + state[i+1, j]*b + state[i, j+1]*b*b + state[i+1, j+1]*b*b*b

    return state_2x2


def init():
    pass


def animate(i):
    ax.cla()
    fig.suptitle(f"Time: {start_trajectories + time_window_duration/2 + i*dt:.2f} s, Max: {ani_sequence[i][1]:.2f} /s", fontsize=12)
    heatmap = ax.imshow(ani_sequence[i][0], cmap='inferno', interpolation='none', vmin=0, vmax=1)
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

vel_sequence = []
while True:
    ret, img = cap.read()
    if not ret:
        break
    i += 1
    print(f"Reading frame {i} of {length}")
    if start_trajectories < (i-1)*dt < end_trajectories:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, of_winsize, 3, 5, 1.2, 0)
        prevgray = gray
        vel_sequence.append(cal_vel(flow))

N = len(vel_sequence)
n = N//2

for occupancy_threshold_speed in occupancy_threshold_speeds:

    state_sequence = []
    count = 0
    for vel in vel_sequence:
        count += 1
        print(f"Mapping state {count} of {N}")
        state = map_to_state(vel, occupancy_threshold_speed)
        state_sequence.append(state)

    if make_animation:

        time_window_size = int(time_window_duration/dt)+1   # in no. of frames
        length = N
        N = time_window_size
        n = N//2
        hi = time_window_size
        lo = 0

        while hi <= length:
            print(f"Getting EP of frame {lo+1} to {hi} of {length}")
            trajectories = np.stack(state_sequence[lo:hi], axis=-1)

            h, w = trajectories.shape[:2]
            ep = np.zeros((h, w))  # entropy production in one time step
            for i, j in np.ndindex(h, w):
                trajectory = trajectories[i, j]
                fh = trajectory[:n]  # first half
                sh = trajectory[n:]  # second half
                shR = np.flip(sh)  # time-reversed second half
                C_fh_shR = cal_cross_parsing_length(fh, shR)
                C_fh_sh = cal_cross_parsing_length(fh, sh)
                sigma = np.log(n)*(C_fh_shR-C_fh_sh)/n
                ep[i, j] = sigma
            epr = ep/dt  # entropy production rate

            epr_avg = np.zeros_like(epr)
            for i, j in np.ndindex(h, w):
                epr_avg[i, j] = (epr[i, j] + epr[i-1, j] + epr[i, j-1] + epr[i-1, j-1])/4
            epr_avg = epr_avg[1:-1, 1:-1]
            max_epr = np.amax(epr_avg)
            epr_avg_normalized = epr_avg/max_epr

            ani_sequence.append([epr_avg_normalized, max_epr])
            lo += 1
            hi += 1

        ani_length = len(ani_sequence)
        ani = animation.FuncAnimation(fig, animate, frames=ani_length, interval=10, repeat=False, init_func=init)
        ani.save(filename[:-4]+"ep_vs_t"+output_filename_suffix+filename[-4:], writer='ffmpeg', fps=fps, dpi=300)

    else:

        trajectories = np.stack(state_sequence, axis=-1)

        count = 0
        h, w = trajectories.shape[:2]
        ep = np.zeros((h, w))  # entropy production in one time step
        total = h*w
        for i, j in np.ndindex(h, w):
            count += 1
            print(f"Getting EP {count} of {total}")
            trajectory = trajectories[i, j]
            fh = trajectory[:n]  # first half
            sh = trajectory[n:]  # second half
            shR = np.flip(sh)  # time-reversed second half
            C_fh_shR = cal_cross_parsing_length(fh, shR)
            C_fh_sh = cal_cross_parsing_length(fh, sh)
            sigma = np.log(n)*(C_fh_shR-C_fh_sh)/n
            ep[i, j] = sigma
        epr = ep/dt  # entropy production rate

        epr_avg = np.zeros_like(epr)
        for i, j in np.ndindex(h, w):
            epr_avg[i, j] = (epr[i, j] + epr[i-1, j] + epr[i, j-1] + epr[i-1, j-1])/4
        epr_avg = epr_avg[1:-1, 1:-1]
        max_epr = np.amax(epr_avg)
        epr_avg_normalized = epr_avg/max_epr
        print(f"\nMax. EPR = {max_epr: .3f} per sec")

        for i in range(1+9*apply_filters):
            filter_level = 0.2*(i/10)
            epr_avg_normalized_filtered = epr_avg_normalized * (epr_avg_normalized > filter_level).astype(int)
            fig, ax = plt.subplots()
            heatmap = ax.imshow(epr_avg_normalized_filtered, cmap='inferno', interpolation='none', vmin=0, vmax=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(heatmap, cax=cax)
            cbar.ax.tick_params(labelsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            plt.savefig(filename[:-4]+f"_ep({filter_level:.2f}up)(v{occupancy_threshold_speed})(max{max_epr:.2f}).png", dpi=300)
            # plt.show()

end_time = time.perf_counter()
print(f"\nTotal run time: {end_time-start_time: .2f} s")
print("\a")
