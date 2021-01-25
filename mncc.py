import numpy as np
from matplotlib import pyplot as plt
import pyfftw

class FFTW:
    def __init__(self, shape, dtype, threads=1, inverse=False):
        self.threads = threads

        if inverse:
            self.arr_in = pyfftw.n_byte_align_empty(shape[0], pyfftw.simd_alignment, dtype=dtype)
            self.fftw = pyfftw.builders.irfft2(self.arr_in, shape[1],
                                               threads=threads, avoid_copy=True)
        else:
            self.arr_in = pyfftw.n_byte_align_empty(shape, pyfftw.simd_alignment, dtype=dtype)
            self.fftw = pyfftw.builders.rfft2(self.arr_in, overwrite_input=True,
                                              threads=threads, avoid_copy=True)
    def get_inverse(self):
        arr_out = self.fftw.get_output_array()
        return FFTW(shape=(arr_out.shape, self.arr_in.shape), dtype=arr_out.dtype, threads=self.threads, inverse=True)

    def __call__(self, arr, reverse=False):
        step = -1 if reverse else 1
        self.arr_in[arr.shape[0]:, :] = 0
        self.arr_in[:arr.shape[0], arr.shape[1]:] = 0
        self.arr_in[:arr.shape[0], :arr.shape[1]] = arr[::step, ::step]
        ret = self.fftw()
        return ret

def _next_regular(target):
    """
    Copied from scipy.

    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & ( target -1)):
        return target

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            p2 = 2** ((quotient - 1).bit_length())

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


class Convolver:
    def __init__(self, shape_first, shape_second, dtype, threads=1):
        self.shape_first = shape_first
        self.shape_second = shape_second

        self.dtype = np.dtype(dtype)

        self.shape_result = np.array(self.shape_first) + self.shape_second - 1
        self.slice_result = tuple([slice(0, int(d)) for d in self.shape_result])
        self.shape_fft = [_next_regular(int(d)) for d in self.shape_result]

        self.fftw = FFTW(shape=self.shape_fft, dtype=self.dtype, threads=threads)
        self.ifftw = self.fftw.get_inverse()

    def FFT(self, img, mask, reverse=False):    
        return [self.fftw(img, reverse).copy(), self.fftw(img**2, reverse).copy(), self.fftw(mask, reverse).copy(),img.shape,np.sum(mask)]
        
    def correlate(self, first, second):
        fft = first * second
        ret = self.ifftw(fft)
        return ret[self.slice_result].copy()

def mncc_fft(convolver,im1,im2, ratio_thresh=0.25): 
    """
    Computes the Normalized Cross Correlation with masked images.

    :param convolver: the convolver object.
    :param im1: Image object: [fft_img, fft_img^2, fft_mask].
    :param im2: Template object.   
    :ratio_thresh: threshold of overlap between image template.
    :param threads: Number of threads (passed to FFTW).
    :return: The resulting cross correlation.

    """       
    b=convolver.correlate(im1[0], im2[2])
    c=convolver.correlate(im1[2], im2[0])
    d = convolver.correlate(im1[2], im2[2]); d[d<1]=1
    nom=convolver.correlate(im1[0], im2[0])-b*c/d;
    denom1=convolver.correlate(im1[1], im2[2])-b**2/d;
    denom2=convolver.correlate(im1[2],im2[1])-c**2/d;
    
    ratio=d/im2[-1]
    if ratio_thresh is None:
        ratio_thresh=0.6*np.nanmax(ratio);
#     print(np.sum(im2[2]),ratio)
    result = nom / np.sqrt(denom1 * denom2)
#     plt.figure(); plt.imshow(result,vmin=0,vmax=1)
    result[ratio<ratio_thresh] = np.nan
#     plt.figure(); plt.imshow(result,vmin=0,vmax=1)

    return result

def mncc(image, tmpl, mask1=None, mask2=None, ratio_thresh=0.25, threads=1):
    """
    Computes the Normalized Cross Correlation with masked images.

    :param image: First array - image.
    :param template: Second array - template.
    :param mask1: Mask for image pixels. Same size as template.    
    :param mask2: Mask for template pixels. Same size as template.
    :param threads: Number of threads (passed to FFTW).
    :return: The resulting cross correlation.

    """
    if mask1 is None:
        mask1 = image<np.inf
    else:
        mask1[np.isnan(image)]=0
    if mask2 is None:
        mask2 = tmpl<np.inf
    else:
        mask2[np.isnan(tmpl)]=0 
    img1 = np.zeros(image.shape,dtype=np.float32); img1[mask1] = image[mask1]; 
    img2 = np.zeros(tmpl.shape,dtype=np.float32); img2[mask2] = tmpl[mask2];  

    convolver = Convolver(image.shape, tmpl.shape, threads=threads, dtype=image.dtype)
    im1=convolver.FFT(img1,mask1)
    im2=convolver.FFT(img2,mask2,reverse=True)
    b=convolver.correlate(im1[0], im2[2])
    c=convolver.correlate(im1[2], im2[0])
    d = convolver.correlate(im1[2], im2[2]); d[d<1]=1
    nom=convolver.correlate(im1[0], im2[0])-b*c/d;
    denom1=convolver.correlate(im1[1], im2[2])-b**2/d;
    denom2=convolver.correlate(im1[2],im2[1])-c**2/d;
    
    ratio=d/np.sum(mask2)

    if ratio_thresh is None:
        ratio_thresh=0.6*np.nanmax(ratio);
#     print(np.sum(mask2),ratio)
    result = nom / np.sqrt(denom1 * denom2)
#     plt.figure(); plt.imshow(result,vmin=0,vmax=1)
    result[ratio<ratio_thresh] = np.nan
#     plt.figure(); plt.imshow(result,vmin=0,vmax=1)

    return result
