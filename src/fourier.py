import numpy as np

class FourierTransform: 
  
  @staticmethod
  def DFT(x):
    # make a numpy array
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)

    return X
  
  @staticmethod
  def inverseDFT(X):
      X = np.asarray(X, dtype=complex)
      N = X.shape[0]
      x = np.zeros(N, dtype=complex)

      for n in range(N):
          for k in range(N):
              x[n] += X[k] * np.exp(2j * np.pi * k * n / N)

          x[n] /= N

      return x
  
  @staticmethod
  def DFT_2D(x):   
    #TODO: refactor 
    x = x.copy()
    x_trans = x.transpose()
    
    x_col_transformed = np.asarray(x_trans, dtype=complex)
    for n, col in enumerate(x_trans):
        x_col_transformed[n] = FourierTransform.DFT(col)
    x_col_transformed = x_col_transformed.transpose()
    
    x_transformed = np.asarray(x, dtype=complex)
    for m, row in enumerate(x_col_transformed):
        x_transformed[m] = FourierTransform.DFT(row)
    return x_transformed

    
  @staticmethod
  def inverseDFT_2D(X):
    
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
      X = np.asarray(X, dtype=complex)
      N = X.shape[0]
      M = X.shape[1]
      x = np.zeros(N, dtype=complex)

      for n in range(N):
          for k in range(N):
              x[n] += X[k] * np.exp(2j * np.pi * k * n / N)

          x[n] /= N

      return x
  
  @staticmethod
  def FFT(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]

    if N % 2 != 0:
        raise ValueError("size of array must be a power of 2")
    elif N <= 32:
        return FourierTransform.DFT(x)
    else:
        even = FourierTransform.FFT(x[::2]) # N/2
        odd = FourierTransform.FFT(x[1::2]) # N/2
        X = np.zeros(N, dtype=complex) # N
        
        n = N//2
        for k in range(N):
          X[k] = even[k % n] + np.exp(-2j * np.pi * k / N) * odd[k % n]
                
        return X
      
  @staticmethod
  def inverseFFT(X):
    """A recursive implementation of the 1D Cooley-Tukey IFFT"""
    X = np.asarray(X, dtype=complex)
    N = X.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return FourierTransform.inverseDFT(X)
    else:
        even = FourierTransform.inverseFFT(X[::2])
        odd = FourierTransform.inverseFFT(X[1::2])
        x = np.zeros(N, dtype=complex)
        
        n = N//2
        for k in range(N):
          x[k] = (n / N) * (even[k % n] + np.exp(2j * np.pi * k / N) * odd[k % n]) 
              
        return x
  
  @staticmethod
  def FFT_2D(x):
    x = x.copy()
    x_trans = x.transpose()
    
    x_col_transformed = np.asarray(x_trans, dtype=complex)
    for n, col in enumerate(x_trans):
        x_col_transformed[n] = FourierTransform.FFT(col)
    x_col_transformed = x_col_transformed.transpose()
    
    x_transformed = np.asarray(x, dtype=complex)
    for m, row in enumerate(x_col_transformed):
        x_transformed[m] = FourierTransform.FFT(row)
    return x_transformed

  
  def inverseFFT_2D(x):
    return 0
  
  @staticmethod
  def test():
      # one dimension
      a = np.random.random(1024)
      fft = np.fft.fft(a)

      # two dimensions
      a2 = np.random.rand(32, 32)
      fft2 = np.fft.fft2(a2)

      tests = (
          (FourierTransform.DFT, a, fft),
          (FourierTransform.inverseDFT, fft, a),
          (FourierTransform.FFT, a, fft),
          (FourierTransform.inverseFFT, fft, a),
          (FourierTransform.DFT_2D, a2, fft2),
          # (DFT.slow_two_dimension_inverse, fft2, a2),
          (FourierTransform.FFT_2D, a2, fft2),
          # (DFT.fast_two_dimension_inverse, fft2, a2)
      )

      for method, args, expected in tests:
          if not np.allclose(method(args), expected):
              print(args)
              print(method(args))
              print(expected)
              raise AssertionError(
                  "{} failed the test".format(method.__name__))
                
  