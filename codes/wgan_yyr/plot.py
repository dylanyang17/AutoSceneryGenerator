import sys
from wgan import WGAN
import numpy as np
import matplotlib.pyplot as plt
from wgan import train_dir

def plot(samples, r, c):
    """
    按照 r * c 的格式绘图, samples 的形状为 (r * c,) + wgan.img_shape
    :return:
    """
    assert samples.shape[0] == r * c

    # 变换到 [0, 255] 上
    samples = np.round((samples + 1) / 2 * 255).astype('uint8')

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(samples[cnt, :, :, :])
            axs[i, j].axis('off')
            cnt += 1
    # fig.savefig(os.path.join(save_dir, 'img.png'))
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Format: python plot.py <epoch index>')
        sys.exit(0)
    epoch = int(sys.argv[1])
    wgan = WGAN()
    wgan.load_model(epoch)
    r = 3
    c = 3
    samples = wgan.choose_best_generated_images(500, r * c)
    plot(samples, r, c)


