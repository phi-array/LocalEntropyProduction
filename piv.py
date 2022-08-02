import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib_scalebar.scalebar import ScaleBar
import cv2
import time


filename = "1_1_60um_1mm_lattice_evolution_4x_GreenRed_SpectraIII_Cyan3_Yellow100_bright.avi"
fps = 20
dt = 0.1666733  # sec
scaling = 1.625  # um/pixel
vec_dist = 25  # no. of pixels between two neighbouring vectors for display purpose
of_winsize = 45  # (15)


cap = cv2.VideoCapture(filename)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
fig, ax = plt.subplots(1, 3, figsize=(19, 8), gridspec_kw={'width_ratios': [20, 20, 1]})


def init():
    pass


def cal_speed(flow):
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    v = np.sqrt(fx*fx+fy*fy)
    v = v*scaling/dt
    return v


def draw_quiver(ax, flow, step=vec_dist):
    h, w = flow.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    ax.quiver(x, y, fx, fy, angles='xy', scale_units='xy', scale=0.15, width=0.002)


def animate(i):
    global prevgray
    ax[0].cla()
    ax[1].cla()
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, of_winsize, 3, 5, 1.2, 0)
    prevgray = gray

    ax[0].imshow(img)
    ax[0].text(0.88, 0.967, f"{(2+i)*dt: .2f} s", transform=ax[0].transAxes, fontsize=14, verticalalignment='top', color='red')
    scalebar = ScaleBar(scaling, "um", fixed_value=300, width_fraction=0.003, location='upper left', border_pad=1.5, scale_loc='top', font_properties={'size': 14}, color='red', box_alpha=0)
    ax[0].add_artist(scalebar)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    piv = ax[1].imshow(cal_speed(flow), cmap='cool', vmin=0, vmax=180, interpolation='none')
    if i == 0:
        cbar = fig.colorbar(piv, cax=ax[2], extend='max')
        cbar.set_label(label="μm/s", size=14)
        cbar.ax.tick_params(labelsize=14)
    draw_quiver(ax[1], flow)
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.subplots_adjust(left=0.02, bottom=0.05, right=0.95, top=0.95, wspace=0.01)
    print(f"frame {2+i} of {length}")


start_time = time.perf_counter()
ani = animation.FuncAnimation(fig, animate, frames=length-1, interval=10, repeat=False, init_func=init)
ani.save(filename[:-4]+"_piv"+filename[-4:], writer='ffmpeg', fps=fps, dpi=300)
# plt.show()
end_time = time.perf_counter()
print(f"\nTotal run time: {end_time-start_time: .2f} s")
print("\a")
