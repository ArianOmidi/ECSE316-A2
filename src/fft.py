from fourier import FourierTransform
import argparse
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from timeit import default_timer as timer


def imageToFFT(img):
    imgFFT = FourierTransform.FFT_2D(resizeImage(img))

    plt.figure(figsize=(15, 5))

    plt.subplot(121), plt.imshow(img, cmap="gray")
    plt.title("Original Image"), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(np.abs(imgFFT), norm=LogNorm())
    plt.colorbar()
    plt.title("Log Scaled 2D FFT"), plt.xticks([]), plt.yticks([])

    plt.suptitle("FFT Transform Result", fontsize=22)
    plt.show()


def denoiseImage(img):
    threshold = 0.1

    # Get FFT of image
    imgFFT = FourierTransform.FFT_2D(resizeImage(img))
    # Get Rows and Columns
    r, c = imgFFT.shape

    imgFFT[int(threshold / 2 * r):int((1 - threshold/2) * r)] = 0
    imgFFT[:, int(threshold / 2 * c):int((1 - threshold/2) * c)] = 0

    print("Number of non-zero: {}".format(int(r * c * threshold)))
    print("Fraction of non-zero: {}".format(threshold))

    denoisedImg = FourierTransform.inverseFFT_2D(imgFFT).real
    denoisedImg = cv2.resize(denoisedImg, (len(img[0]), len(img)))

    # Plot values
    plt.figure(figsize=(15, 5))
    plt.subplot(121), plt.imshow(img, cmap="gray")
    plt.title("Original Image"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(denoisedImg, cmap="gray")
    plt.title("Denoised Image"), plt.xticks([]), plt.yticks([])
    plt.suptitle("Threshold: {}%".format(100 * threshold), fontsize=22)
    plt.show()


def compressImage(img):
    return 0


def plotRuntime():
    tests = [np.random.random((2 ** 5, 2 ** 5)),
             np.random.random((2 ** 6, 2 ** 6)),
             np.random.random((2 ** 7, 2 ** 7)),
             np.random.random((2 ** 8, 2 ** 8))]

    # Sample 10 values DFT and FFT
    for arr in tests:
        DFT_vals = []
        FFT_vals = []
        for i in range(1, 10):
            # Slow Algorithm Timing
            start = timer()
            FourierTransform.DFT_2D(arr)
            end = timer()
            DFT_vals.append(end - start)

            # Fast Algorithm Timing
            start = timer()
            FourierTransform.FFT_2D(arr)
            end = timer()
            FFT_vals.append(end - start)

        print('================  DFT ========================')
        print("DFT Mean: ", np.mean(DFT_vals))
        print("DFT Variance: ", np.var(DFT_vals))

        print('================  FFT  ========================')
        print("FFT Mean: ", np.mean(FFT_vals))
        print("FFT Variance: ", np.var(FFT_vals))
        print('===============================================')


def resizeImage(img):
    if (len(img[0]) == 2 ** (int(np.log2(len(img[0]))))):
        width = len(img[0])
    else:
        width = 2 ** int((np.log2(len(img[0]))) + 1)

    if (len(img) == 2 ** (int(np.log2(len(img))))):
        height = len(img)
    else:
        height = 2 ** int((np.log2(len(img))) + 1)

    return cv2.resize(img, (width, height))


def main():
    parser = argparse.ArgumentParser(description='Compute Fourier Transforms')
    parser.add_argument('-m', type=int, default=1,
                        action='store', help='mode to be selected')
    parser.add_argument('-i', type=str, default='moonlanding.png',
                        action='store', help='image filename')

    args = parser.parse_args()
    img = args.i
    mode = args.m

    # Get image
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    if mode == 1:
        imageToFFT(img)
    elif mode == 2:
        denoiseImage(img)
    elif mode == 3:
        compressImage(img)
    elif mode == 4:
        plotRuntime()
    return 0


if __name__ == "__main__":
    main()
