from fourier import FourierTransform
import argparse
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import statistics as stats
from timeit import default_timer as timer


def imageToFFT(img):
    imgFFT = FourierTransform.FFT_2D(resizeImage(img))

    # TODO: refactor
    plt.figure(figsize=(15, 5))

    plt.subplot(121), plt.imshow(img, cmap="gray")
    plt.title("Original Image"), plt.xticks([]), plt.yticks([])

    plt.subplot(123), plt.imshow(np.abs(imgFFT), norm=LogNorm())
    plt.colorbar()
    plt.title("Log Scaled 2D FFT"), plt.xticks([]), plt.yticks([])

    plt.suptitle("FFT Transform Result", fontsize=22)
    plt.show()


def denoiseImage(img):
    threshold = 0.3

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

    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(121), plt.imshow(img, cmap="gray")
    plt.title("Original Image"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(denoisedImg, cmap="gray")
    plt.title("Denoised Image"), plt.xticks([]), plt.yticks([])
    plt.suptitle("Threshold: {}%".format(100 * threshold), fontsize=22)
    plt.show()


def compressImage(img):
    transformed_original = dft2_fast(img)
    num_non_zeros = np.count_nonzero(transformed_original)
    print("original: num zeros: ", num_non_zeros)
    data_csr = sparse.csr_matrix(transformed_original)
    sparse.save_npz("original.npz", data_csr)
    size = os.path.getsize("original.npz")
    plt.figure(figsize=(15, 5))
    plt.subplot(231), plt.imshow(img, cmap="gray")
    plt.title("Original, size={} bytes, non zeros={}".format(
        size, num_non_zeros)), plt.xticks([]), plt.yticks([])
    compression_factors = [30, 60, 80, 90, 95]
    index_count = 2
    for factor in compression_factors:
        transformed = transformed_original.copy()
        thresh = np.percentile(abs(transformed), factor)
        transformed[abs(transformed) < thresh] = 0
        num_non_zeros = np.count_nonzero(transformed)
        data_csr = sparse.csr_matrix(transformed)
        file_name = "@{}%.npz".format(factor)
        sparse.save_npz(file_name, data_csr)
        size = os.path.getsize(file_name)
        back = inverse_dft2_fast(transformed).real
        plt.subplot(2, 3, index_count), plt.imshow(back, cmap="gray")
        plt.title("@ {}%, size={} bytes, non zeros={}".format(factor,
                  size, num_non_zeros)), plt.xticks([]), plt.yticks([])
        print(factor, "%: num non zeros: ", num_non_zeros)
        index_count = index_count + 1
    plt.suptitle("Compression Levels", fontsize=22)
    plt.show()


def plotRuntime():
    arr1 = np.random.random((2 ** 5, 2 ** 5))
    arr2 = np.random.random((2 ** 6, 2 ** 6))
    arr3 = np.random.random((2 ** 7, 2 ** 7))
    arr4 = np.random.random((2 ** 8, 2 ** 8))
    array_dict = {'2^5 x 2^5': arr1, '2^6 x 2^6': arr2,
                  '2^7 x 2^7': arr3, '2^8 x 2^8': arr4}

    plt.figure(figsize=(15, 5))
    plt.title('Discrete Time Fourier Transform Runtime Analysis')
    plt.xlabel('Problem Size (Array Dimensions)')
    plt.ylabel('Average Runtime (sec)')

    x_axis = ['2^5 x 2^5', '2^6 x 2^6', '2^7 x 2^7', '2^8 x 2^8']
    y_axis_slow = []
    y_axis_fast = []
    confidence_interval_slow = []
    confidence_interval_fast = []

    # Measure Slow & Fast Algorithm, Take Avg of 10 Readings
    for arr_name, arr in array_dict.items():
        slow_duration_readings = []
        fast_duration_readings = []
        for i in range(1, 10):
            # Slow Algorithm Timing
            start = timer()
            FourierTransform.DFT_2D(arr)
            end = timer()
            slow_duration_readings.append(end - start)

            # Fast Algorithm Timing
            start = timer()
            FourierTransform.FFT_2D(arr)
            end = timer()
            fast_duration_readings.append(end - start)

        print('================  Slow Algorithm ========================')
        avg_slow_duration = sum(slow_duration_readings) / 10
        y_axis_slow.append(avg_slow_duration)
        confidence_interval_slow.append(
            stats.stdev(slow_duration_readings) * 2)
        print("Slow Algorithm Mean: ", np.mean(slow_duration_readings))
        print("Slow Algorithm Variance: ", np.var(slow_duration_readings))

        print('================  Fast Algorithm ========================')
        avg_fast_duration = sum(fast_duration_readings) / 10
        y_axis_fast.append(avg_fast_duration)
        confidence_interval_fast.append(
            stats.stdev(fast_duration_readings) * 2)
        print("Fast Algorithm Mean: ", np.mean(fast_duration_readings))
        print("Fast Algorithm Variance: ", np.var(fast_duration_readings))
        print('=========================================================')

    # Plot w/ Error Bars (Stdev * 2)
    plt.errorbar(x=x_axis, y=y_axis_slow,
                 yerr=confidence_interval_slow, label='naive')
    plt.errorbar(x=x_axis, y=y_axis_fast,
                 yerr=confidence_interval_fast, label='fast')
    plt.legend(loc='upper left', numpoints=1)
    plt.show()


def resizeImage(img):
  # TODO: refactor
    width = len(img[0])
    height = len(img)
    width = width if width == 2 ** (int(np.log2(width))
                                    ) else 2 ** (int(np.log2(width)) + 1)
    height = height if height == 2 ** (int(np.log2(height))
                                       ) else 2 ** int((np.log2(height)) + 1)

    return cv2.resize(img, (width, height))


def main():
    # TODO: refactor
    parser = argparse.ArgumentParser(description='Compute Fourier Transforms')
    parser.add_argument('-m', type=int, default=1,
                        action='store', help='mode to be selected')
    parser.add_argument('-i', type=str, default='moonlanding.png',
                        action='store', help='image filename')
    parser.add_argument('--test', type=bool, default=False,
                        action='store', help='run tests')

    args = parser.parse_args()
    img = args.i
    mode = args.m
    test = args.test

    if (test):
        FourierTransform.test()
        return 0

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
