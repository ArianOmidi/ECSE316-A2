import numpy as np
import matplotlib


class DFT:
    @staticmethod
    def slow_one_dimension(a):
        a = np.asarray(a, dtype=complex)
        N = a.shape[0]
        res = np.zeros(N, dtype=complex)

        for k in range(N):
            for n in range(N):
                res[k] += a[n] * np.exp(-2j * np.pi * k * n / N)

        return res

    @staticmethod
    def slow_one_dimension_inverse(a):
        a = np.asarray(a, dtype=complex)
        N = a.shape[0]
        res = np.zeros(N, dtype=complex)

        for n in range(N):
            for k in range(N):
                res[n] += a[k] * np.exp(2j * np.pi * k * n / N)

            res[n] /= N

        return res

    @staticmethod
    def fast_one_dimension(a):
        a = np.asarray(a, dtype=complex)
        N = a.shape[0]

        if N % 2 != 0:
            raise AssertionError("size of a must be a power of 2")
        elif N <= 16:
            return DFT.slow_one_dimension(a)
        else:
            even = DFT.fast_one_dimension(a[::2])
            odd = DFT.fast_one_dimension(a[1::2])
            res = np.zeros(N, dtype=complex)

            half_size = N//2
            for n in range(N):
                res[n] = even[n % half_size] + \
                    np.exp(-2j * np.pi * n / N) * odd[n % half_size]

            return res

    @staticmethod
    def fast_one_dimension_inverse(a):
        a = np.asarray(a, dtype=complex)
        N = a.shape[0]

        if N % 2 != 0:
            raise AssertionError("size of a must be a power of 2")
        elif N <= 16:
            return DFT.slow_one_dimension_inverse(a)
        else:
            even = DFT.fast_one_dimension_inverse(a[::2])
            odd = DFT.fast_one_dimension_inverse(a[1::2])
            res = np.zeros(N, dtype=complex)

            half_size = N//2
            for n in range(N):
                res[n] = half_size * even[n % half_size] + \
                    np.exp(2j * np.pi * n / N) * half_size * odd[n % half_size]
                res[n] /= N

            return res

    @staticmethod
    def slow_two_dimension(a):
        a = np.asarray(a, dtype=complex)
        N, M = a.shape
        res = np.zeros((N, M), dtype=complex)

        for k in range(N):
            for l in range(M):
                for m in range(M):
                    for n in range(N):
                        res[k, l] += a[n, m] * \
                            np.exp(-2j * np.pi * ((l * m / M) + (k * n / N)))

        return res

    @staticmethod
    def slow_two_dimension_inverse(a):
        a = np.asarray(a, dtype=complex)
        N, M = a.shape
        res = np.zeros((N, M), dtype=complex)

        for k in range(N):
            for l in range(M):
                for m in range(M):
                    for n in range(N):
                        res[k, l] += a[n, m] * \
                            np.exp(2j * np.pi * ((l * m / M) + (k * n / N)))
                res[k, l] /= M * N

        return res

    @staticmethod
    def fast_two_dimension(a):
        a = np.asarray(a, dtype=complex)
        N, M = a.shape
        res = np.zeros((N, M), dtype=complex)

        for col in range(M):
            res[:, col] = DFT.fast_one_dimension(a[:, col])

        for row in range(N):
            res[row, :] = DFT.fast_one_dimension(res[row, :])

        return res

    @staticmethod
    def fast_two_dimension_inverse(a):
        a = np.asarray(a, dtype=complex)
        N, M = a.shape
        res = np.zeros((N, M), dtype=complex)

        for row in range(N):
            res[row, :] = DFT.fast_one_dimension_inverse(a[row, :])

        for col in range(M):
            res[:, col] = DFT.fast_one_dimension_inverse(res[:, col])

        return res

    @staticmethod
    def test():
        # one dimension
        a = np.random.random(1024)
        fft = np.fft.fft(a)

        # two dimensions
        a2 = np.random.rand(32, 32)
        fft2 = np.fft.fft2(a2)

        tests = (
            (DFT.slow_one_dimension, a, fft),
            (DFT.slow_one_dimension_inverse, fft, a),
            (DFT.fast_one_dimension, a, fft),
            (DFT.fast_one_dimension_inverse, fft, a),
            (DFT.slow_two_dimension, a2, fft2),
            (DFT.slow_two_dimension_inverse, fft2, a2),
            (DFT.fast_two_dimension, a2, fft2),
            (DFT.fast_two_dimension_inverse, fft2, a2)
        )

        for method, args, expected in tests:
            if not np.allclose(method(args), expected):
                print(args)
                print(method(args))
                print(expected)
                raise AssertionError(
                    "{} failed the test".format(method.__name__))
