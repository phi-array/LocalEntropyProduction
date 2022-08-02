import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
import cv2
import time


filename = "1_1_60um_1mm_lattice_evolution_4x_GreenRed_SpectraIII_Cyan3_Yellow100_low_res(1d25).m4v"
dt = 0.1666733  # sec
scaling = 1.625  # um/pixel
start_trajectories = 14  # time in sec when the trajectories begin (14)
end_trajectories = 400  # time in sec when the trajectories end (400)


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
        y = y + chr(elem+33)
    for elem in Z:
        z = z + chr(elem+33)
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


def map_to_state(vel):

    orientations = []
    orientations.append((vel[:, :, 0] >= 0) & (vel[:, :, 0] <= np.pi/4) & (vel[:, :, 0] > np.pi*7/4) & (vel[:, :, 0] <= np.pi*2))  # left
    orientations.append((vel[:, :, 0] > np.pi/4) & (vel[:, :, 0] <= np.pi*3/4))    # down
    orientations.append((vel[:, :, 0] > np.pi*3/4) & (vel[:, :, 0] <= np.pi*5/4))  # right
    orientations.append((vel[:, :, 0] > np.pi*5/4) & (vel[:, :, 0] <= np.pi*7/4))  # up

    speeds = []
    speeds.append((vel[:, :, 1] > 1))
#    speeds.append((vel[:, :, 1] > 1) & (vel[:, :, 1] <= 50))
#    speeds.append((vel[:, :, 1] > 50) & (vel[:, :, 1] <= 100))
#    speeds.append((vel[:, :, 1] > 100) & (vel[:, :, 1] <= 150))
#    speeds.append((vel[:, :, 1] > 150) & (vel[:, :, 1] <= 200))
#    speeds.append((vel[:, :, 1] > 200))

    state = np.zeros_like(vel[:, :, 0], dtype=int)
    b = 1
    for speed in speeds:
        for orientation in orientations:
            state += b*(speed & orientation).astype(int)
            b += 1

    return state


vel_sequence = []
state_sequence = []

start_time = time.perf_counter()

cap = cv2.VideoCapture(filename)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
i = 1

while True:
    ret, img = cap.read()
    if not ret:
        break
    i += 1
    print(f"Reading frame {i} of {length}")
    if start_trajectories < (i-1)*dt < end_trajectories:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray
        vel_sequence.append(cal_vel(flow))

N = len(vel_sequence)
n = N//2

count = 0
for vel in vel_sequence:
    count += 1
    print(f"Translating state {count} of {N}")
    state = map_to_state(vel)
    state_sequence.append(state)

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

fig, ax = plt.subplots()
heatmap = ax.imshow(epr, cmap='inferno', interpolation='none', vmin=0)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(heatmap, cax=cax)
cbar.ax.tick_params(labelsize=12)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
plt.savefig(filename[:-4]+"_ep.png", dpi=300)
# plt.show()

end_time = time.perf_counter()
print(f"\nTotal run time: {end_time-start_time: .2f} s")
print("\a")
