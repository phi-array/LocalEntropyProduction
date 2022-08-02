import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import time


# Settings:
fps = 20
output_format = ".m4v"
output_filename_prefix = "compare_(1d16)vs_t(tw25s)"

video_list = []
video_list.append(["1_1_60um_1mm_lattice_evolution_4x_GreenRed_SpectraIII_Cyan3_Yellow100_bright_low_res(1d16)_vorticity_sum_normalized_abs_vs_t(tw25s).m4v", "Absolute vorticity sum (normalized)", "avs"])
video_list.append(["1_1_60um_1mm_lattice_evolution_4x_GreenRed_SpectraIII_Cyan3_Yellow100_low_res(1d16)ep_vs_t(tw25s)(or8).m4v", "Entropy production rate (normalized)", "epr"])


def init():
    pass


def animate(i):
    for ax in axs:
        ax.cla()
    for index, cap in enumerate(cap_list):
        ret, img = cap.read()
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[index].imshow(frame)
        axs[index].set_xlabel(video_list[index][1], fontsize=18)
        axs[index].set_xticks([])
        axs[index].set_yticks([])
        axs[index].spines['top'].set_visible(False)
        axs[index].spines['right'].set_visible(False)
        axs[index].spines['bottom'].set_visible(False)
        axs[index].spines['left'].set_visible(False)
    plt.subplots_adjust(left=0, bottom=0.055, right=1, top=1, wspace=0)
    print(f"frame {i+1} of {length}")


# Main:
start_time = time.perf_counter()

cap_list = []
for video in video_list:
    cap_list.append(cv2.VideoCapture(video[0]))
length = int(cap_list[0].get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap_list[0].get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_list[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
video_count = len(cap_list)
fig, axs = plt.subplots(1, video_count, figsize=(8*(width/height)*video_count, 8))

ani = animation.FuncAnimation(fig, animate, frames=length, interval=10, repeat=False, init_func=init)
ani.save(output_filename_prefix + "".join("_"+x[2] for x in video_list) + output_format, writer='ffmpeg', fps=fps, dpi=300)
# plt.show()

end_time = time.perf_counter()
print(f"\nTotal run time: {end_time-start_time: .2f} s")
print("\a")
