#!/usr/bin/env python3
"""CSc 8830 Module 3: spatial and Fourier-domain image blurring.

Install: python -m pip install -r requirements.txt
Run:     python module3.py
Custom:  python module3.py --image my_image.png --kernel-size 15 --sigma 2.5
Outputs are written beside this script unless --out is supplied. The default
input is a deterministic mathematical test image, not a camera photograph.
This script computes and validates the results, saves the arrays and figures,
and writes the PDF report. It performs no uploads or Classroom submissions.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import textwrap
import time

import numpy as np
import scipy
from scipy.signal import convolve2d
from PIL import Image
import PIL
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
REPO = 'https://github.com/aabdelkarem/csc8830-module3-image-blurring'
ASSIGNMENT = 'https://docs.google.com/document/d/1KQo_l3bnipHFNhmJOHwoMuEz9xKV-I1FBO1RP9MCWVA/edit'
BLUE, ORANGE, INK = '#183e50', '#ac542b', '#20313a'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def test_image():
    """Edges, repeated texture and a disk make smoothing easy to inspect."""
    yy, xx = np.mgrid[0:192, 0:256]
    image = .15 + .6 * ((xx // 16 + yy // 16) % 2).astype(float)
    image[(xx - 170)**2 + (yy - 85)**2 < 36**2] = 1.
    image[20:60, 25:100] = .3
    return image


def gaussian_kernel(size=15, sigma=2.5):
    if size < 3 or size % 2 != 1 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Use an odd kernel size >= 3 and a finite positive sigma.')
    v = np.arange(size, dtype=float) - size // 2
    kernel = np.exp(-(v[:, None]**2 + v[None, :]**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def fourier_convolution(image, kernel):
    """Full LINEAR convolution: zero-pad each axis to image+kernel-1."""
    shape = tuple(a + b - 1 for a, b in zip(image.shape, kernel.shape))
    image_fft = np.fft.fft2(image, s=shape, axes=(-2, -1))
    kernel_fft = np.fft.fft2(kernel, s=shape, axes=(-2, -1))
    product = image_fft * kernel_fft
    complex_output = np.fft.ifft2(product)
    return complex_output.real, image_fft, kernel_fft, product, complex_output.imag


def direct_dft2(image, inverse=False):
    """Small independent DFT using explicit exponential matrices, not FFT."""
    p, q = image.shape
    sign = 1 if inverse else -1
    rows = np.exp(sign * 2j * np.pi * np.outer(np.arange(p), np.arange(p)) / p)
    cols = np.exp(sign * 2j * np.pi * np.outer(np.arange(q), np.arange(q)) / q)
    result = rows @ image @ cols.T
    return result / (p * q) if inverse else result


def run_experiment(image=None, kernel_size=15, sigma=2.5, progress=None):
    """Execute both algorithms; progress receives actual computed states."""
    def emit(name, data):
        if progress is not None:
            progress(name, data)

    if image is None:
        a = test_image()
        origin = 'Deterministic mathematical test pattern; not a camera capture.'
        source_hash = None
    else:
        path = Path(image)
        a = np.asarray(Image.open(path).convert('L'), dtype=np.float64) / 255.
        origin = 'Provided image converted to grayscale and normalized to [0,1].'
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if min(a.shape) <= kernel_size:
        raise ValueError('Image dimensions must exceed the kernel size.')
    k = gaussian_kernel(kernel_size, sigma)
    emit('input', {'image': a, 'kernel': k, 'origin': origin})

    # scipy.signal.convolve2d directly evaluates spatial convolution.
    start = time.perf_counter()
    spatial = convolve2d(a, k, mode='full', boundary='fill', fillvalue=0)
    spatial_time = time.perf_counter() - start
    pad = kernel_size // 2
    crop = (slice(pad, pad + a.shape[0]), slice(pad, pad + a.shape[1]))
    emit('spatial', {'image': a, 'output': spatial[crop], 'full_shape': spatial.shape})

    start = time.perf_counter()
    frequency, f, h, product, imaginary = fourier_convolution(a, k)
    fft_time = time.perf_counter() - start
    emit('frequency', {'image_fft': f, 'kernel_fft': h, 'product': product,
                       'output': frequency[crop], 'full_shape': frequency.shape})

    difference = spatial - frequency
    maximum = float(np.max(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference**2)))
    np.testing.assert_allclose(frequency, spatial, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(spatial[crop], convolve2d(a, k, mode='same'), atol=1e-11)
    emit('comparison', {'spatial': spatial[crop], 'fourier': frequency[crop],
                        'difference': np.abs(difference[crop]), 'max_error': maximum,
                        'rmse': rmse})

    # A correctly centered, input-size transform gives CIRCULAR convolution.
    # This counterexample changes only boundary handling, not kernel alignment.
    embedded = np.zeros_like(a)
    embedded[:kernel_size, :kernel_size] = k
    centered = np.roll(embedded, (-pad, -pad), axis=(0, 1))
    circular = np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(centered)).real
    circular_reference = convolve2d(a, k, mode='same', boundary='wrap')
    np.testing.assert_allclose(circular, circular_reference, atol=1e-11)
    interior_error = float(np.max(np.abs((circular - spatial[crop])[pad:-pad, pad:-pad])))
    np.testing.assert_allclose(circular[pad:-pad, pad:-pad],
                               spatial[crop][pad:-pad, pad:-pad], atol=1e-11)
    boundary_error = float(np.max(np.abs(circular - spatial[crop])))
    emit('padding', {'zero_boundary': spatial[crop], 'circular': circular,
                     'difference': np.abs(circular-spatial[crop]),
                     'max_error': boundary_error, 'interior_error': interior_error})

    # A symmetric blur kernel alone cannot expose convolution/correlation flips.
    rng = np.random.default_rng(8830)
    random_image = rng.normal(size=(9, 11))
    asymmetric = np.array([[1., 2., -1.], [0., 4., 3.]])
    asymmetric_error = float(np.max(np.abs(fourier_convolution(random_image, asymmetric)[0]
                                  - convolve2d(random_image, asymmetric, mode='full'))))
    assert asymmetric_error < 1e-11
    impulse = np.zeros((7, 8)); impulse[0, 0] = 1
    impulse_result = fourier_convolution(impulse, asymmetric)[0]
    expected_impulse = np.zeros_like(impulse_result)
    expected_impulse[:2, :3] = asymmetric
    impulse_error = float(np.max(np.abs(impulse_result - expected_impulse)))
    assert impulse_error < 1e-11
    sum_error = float(abs(spatial.sum() - a.sum() * k.sum()))
    np.testing.assert_allclose(spatial.sum(), a.sum() * k.sum(), rtol=1e-12)

    # Hand-checkable example, also checked by the independent explicit DFT.
    small = np.array([[1., 2.], [3., 4.]])
    box = np.ones((2, 2)) / 4
    expected = np.array([[.25, .75, .5], [1., 2.5, 1.5], [.75, 1.75, 1.]])
    ap, kp = np.zeros((3, 3)), np.zeros((3, 3))
    ap[:2, :2], kp[:2, :2] = small, box
    manual_dft = direct_dft2(direct_dft2(ap) * direct_dft2(kp), inverse=True)
    small_error = float(np.max(np.abs(manual_dft.real - expected)))
    np.testing.assert_allclose(convolve2d(small, box, mode='full'), expected, atol=1e-12)
    np.testing.assert_allclose(manual_dft, expected, atol=1e-12)
    example = {'image': small.tolist(), 'kernel': box.tolist(),
               'expected_full_convolution': expected.tolist(),
               'explicit_dft_max_abs_error': small_error,
               'center_calculation': '(1 + 2 + 3 + 4) / 4 = 2.5'}
    emit('example', {'image': small, 'kernel': box, 'expected': expected, 'error': small_error})
    checks = {'full_spatial_fft_agreement': maximum < 1e-11,
              'same_crop_alignment': True, 'asymmetric_kernel': asymmetric_error < 1e-11,
              'impulse_origin': impulse_error < 1e-11, 'full_output_sum': True,
              'explicit_dft_example': small_error < 1e-12,
              'circular_boundary_reference': True, 'interior_boundary_agreement': True}
    metrics = {'input_provenance': origin, 'source_file_sha256': source_hash,
               'input_shape': list(a.shape), 'kernel_shape': list(k.shape),
               'kernel_sigma_pixels': sigma, 'kernel_sum': float(k.sum()),
               'fft_and_full_output_shape': list(spatial.shape),
               'max_abs_error': maximum, 'rmse': rmse,
               'max_imaginary_roundoff': float(np.max(np.abs(imaginary))),
               'circular_vs_zero_boundary_max_error': boundary_error,
               'circular_vs_zero_interior_max_error': interior_error,
               'asymmetric_kernel_max_error': asymmetric_error,
               'impulse_max_error': impulse_error, 'full_output_sum_error': sum_error,
               'checks': checks, 'all_checks_passed': all(checks.values()),
               'single_run_seconds_not_a_benchmark': {'spatial': spatial_time, 'fft': fft_time}}
    emit('validation', metrics)
    return {'image': a, 'kernel': k, 'spatial': spatial, 'fourier': frequency,
            'difference': difference, 'image_fft': f, 'kernel_fft': h, 'product': product,
            'circular': circular, 'crop': crop, 'metrics': metrics, 'example': example}


def make_figures(result, out):
    a, k, crop = result['image'], result['kernel'], result['crop']
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.0))
    fields = [a, result['spatial'][crop], result['fourier'][crop],
              np.abs(result['difference'][crop]),
              np.log1p(np.abs(np.fft.fftshift(result['image_fft']))),
              np.abs(np.fft.fftshift(result['kernel_fft']))]
    labels = ['Input test image', 'Spatial convolution', 'Fourier equivalent',
              'Absolute difference (magnified scale)', 'log(1 + |image DFT|)', '|kernel DFT|']
    for i, (ax, field, title) in enumerate(zip(axes.flat, fields, labels)):
        im = ax.imshow(field, cmap='magma' if i == 3 else 'gray',
                       **({'vmin': 0, 'vmax': 1} if i < 3 else {}))
        ax.set_title(title, fontsize=11); ax.axis('off')
        if i == 3: fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.3))
    for ax, data, title in zip(axes, [result['spatial'][crop], result['circular'],
                                       np.abs(result['circular']-result['spatial'][crop])],
                              ['Zero-extension blur', 'Circular (wrap) boundary', 'Boundary mismatch']):
        ax.imshow(data, cmap='gray', vmin=0, vmax=1); ax.set_title(title, fontsize=11); ax.axis('off')
    fig.tight_layout(); fig.savefig(out/'padding_comparison.png', dpi=150); plt.close(fig)
    Image.fromarray(np.rint(a*255).astype('uint8')).save(out/'input_test_image.png')


def page(title, subtitle, number):
    f = plt.figure(figsize=(8.27, 11.69), facecolor='white')
    f.text(.075, .938, title, fontsize=20, color=BLUE, weight='bold')
    f.text(.075, .908, subtitle, fontsize=10, color=ORANGE)
    f.text(.075, .045, 'Aya Abdelkarem | CSc 8830 | Module 3', fontsize=8, color=BLUE)
    f.text(.92, .045, str(number), fontsize=8, color=BLUE)
    return f


def paragraph(f, text, y, size=10.2, width=96, step=.022):
    for block in text.split('\n'):
        for line in textwrap.wrap(block, width) or ['']:
            f.text(.075, y, line, fontsize=size, color=INK); y -= step
        y -= .008
    return y


def put_picture(f, path, bounds):
    ax = f.add_axes(bounds); ax.imshow(Image.open(path)); ax.axis('off')


def make_report(result, out):
    m = result['metrics']
    height, width = m['input_shape']
    size = m['kernel_shape'][0]
    radius = size // 2
    full_h, full_w = m['fft_and_full_output_shape']
    origin = ('The supplied grayscale image is a deterministic mathematical test pattern with texture, a disk and sharp edges. It is generated by the program; no camera capture is claimed.' if m['source_file_sha256'] is None else 'The supplied image is converted to grayscale. Its original-file SHA-256 is recorded in metrics.json.')
    with PdfPages(out/'Module_3_Report.pdf', metadata={'Title':'Module 3: Image blurring and the convolution theorem',
                                                     'Author':'Aya Abdelkarem'}) as pdf:
        f = page('Module 3: Image blurring', 'Spatial filtering and its Fourier-domain equivalent', 1)
        y = paragraph(f, 'Objective. Implement image blurring and demonstrate that spatial convolution gives the same output as multiplication of the image and filter transforms in the Fourier domain.', .855)
        f.text(.075, y-.008, 'GitHub repository (source, data and demonstration)', fontsize=10.2, weight='bold', color=BLUE)
        f.text(.075, y-.037, REPO, fontsize=8.5, color='#166a9b', url=REPO)
        y = paragraph(f, f'Input. {origin} Dimensions: {height} x {width}; values use float64 on [0,1].\nFilter. A finite {size} x {size} Gaussian kernel with sigma = {m["kernel_sigma_pixels"]:g} pixels is normalized to sum to one. Both algorithms use this same kernel. The centered kernel weights nearby pixels most strongly; r = {radius}.', y-.10)
        f.text(.11, y-.01, r'$h[i,j]=\frac{\exp(-(i^2+j^2)/(2\sigma^2))}{\sum_{u=-r}^{r}\sum_{v=-r}^{r}\exp(-(u^2+v^2)/(2\sigma^2))}$', fontsize=14, color=BLUE)
        y = paragraph(f, f'Spatial method. scipy.signal.convolve2d evaluates full linear convolution using zero-valued pixels outside the image. The output is ({height}+{size}-1) x ({width}+{size}-1) = {full_h} x {full_w}.\nFourier method. Pad both arrays to {full_h} x {full_w}, compute their two-dimensional FFTs, multiply the complex coefficients point by point, and inverse transform. Retain the real component after checking that the imaginary remainder is numerical round-off.\nComparison. Compare every pixel of the full outputs before rounding, clipping or saving display images. Pictures use the identical crop [{radius}:{radius+height}, {radius}:{radius+width}].', y-.11)
        paragraph(f, 'Why blurring occurs. A normalized Gaussian averages local intensities. In frequency terms, its transform attenuates high-frequency variation. Fine checkerboard transitions and sharp object edges therefore become smoother. Zero extension also darkens the outer border; that is the chosen boundary model.', y-.02)
        pdf.savefig(f); plt.close(f)

        f = page('Why the two methods agree', 'Convolution theorem and a typed worked example', 2)
        y = paragraph(f, 'Let f be an H x W image and h a Kh x Kw kernel, with both zero outside their finite support. Choose P >= H+Kh-1 and Q >= W+Kw-1. This prevents a periodic DFT convolution from wrapping nonzero values into the linear-convolution support.', .855)
        equations = [r'$g[m,n]=\sum_a\sum_b f[a,b]h[m-a,n-b]$',
                     r'$G[k,\ell]=\sum_m\sum_n g[m,n]e^{-2\pi i(km/P+\ell n/Q)}$',
                     r'$G[k,\ell]=\sum_a\sum_b f[a,b]\sum_r\sum_s h[r,s]$',
                     r'$\qquad\qquad\times e^{-2\pi i(k(a+r)/P+\ell(b+s)/Q)}$',
                     r'$G[k,\ell]=F[k,\ell]H[k,\ell]$',
                     r'$g=\mathrm{IDFT}\{F H\}$']
        for equation in equations:
            f.text(.10, y-.015, equation, fontsize=13.5, color=BLUE); y -= .060
        y = paragraph(f, 'Substitute the convolution into the DFT and set r=m-a and s=n-b. The exponential separates into a factor involving a,b and a factor involving r,s. The sums then factor into F and H. NumPy uses the matching inverse normalization, 1/(PQ), so no additional scale factor is required.', y-.02, size=10, width=99)
        f.text(.075, y-.005, 'Small example: 2 x 2 image and averaging kernel', fontsize=11, weight='bold', color=BLUE)
        # Matrices are rendered as monospaced typed arithmetic, not photographed handwriting.
        y -= .055
        for x, title, matrix in [( .09, 'f', [[1,2],[3,4]]), (.35, 'h', [[.25,.25],[.25,.25]]),
                                  (.65, 'f * h', [[.25,.75,.5],[1,2.5,1.5],[.75,1.75,1]])]:
            f.text(x, y, title, fontsize=11, color=BLUE, weight='bold')
            for j,row in enumerate(matrix):
                f.text(x, y-.033*(j+1), '['+'  '.join(f'{v:g}' for v in row)+']', family='monospace', fontsize=10)
        paragraph(f, 'Center: (1+2+3+4)/4 = 2.5. Top-left: 1/4 = 0.25. The full result sums to 10, matching the image sum because the kernel sums to one. An explicit 3 x 3 DFT, evaluated with exponential matrices independently of NumPy FFT, also reproduces this table.', y-.15, size=9.4, width=104, step=.020)
        pdf.savefig(f); plt.close(f)

        f = page('Executed image results', 'Identical full-output comparison before display conversion', 3)
        paragraph(f, f"Maximum absolute difference: {m['max_abs_error']:.16g}\nRoot mean squared difference: {m['rmse']:.16g}\nMaximum imaginary residual after inverse FFT: {m['max_imaginary_roundoff']:.6g}\nFull output: {full_h} x {full_w} pixels; input-size display: {height} x {width} pixels.", .855, size=10.4)
        put_picture(f, out/'comparison.png', [.055,.34,.89,.40])
        paragraph(f, 'The spatial and Fourier blurred images agree to float64 round-off. The difference panel has an automatically magnified color scale, so visible structure in that panel does not imply a perceptible image discrepancy. The comparison uses normalized floating-point intensities, not rounded 8-bit saved PNGs.\nThe image spectrum is displayed as log(1+|F|); the filter spectrum is |H|. fftshift is used only to center spectra for visualization. The numerical convolution multiplies the unshifted transforms, preserving the correct spatial origin.\nThe complete input, kernel, both full outputs, complex transforms and difference array are saved in convolution_arrays.npz. Exact numerical metrics and the eight validation results are in metrics.json.', .29, size=10, width=99)
        pdf.savefig(f); plt.close(f)

        f = page('Validation and reproduction', 'Boundary conditions matter; numerical agreement is explicitly checked', 4)
        paragraph(f, 'Eight checks passed: full-output agreement; centered-crop alignment; asymmetric-kernel agreement; impulse origin; conservation of the full-output sum; an independent explicit-DFT example; agreement with SciPy circular boundaries; and interior agreement between circular and zero boundaries.', .855, size=10, width=99)
        paragraph(f, f"Asymmetric kernel maximum error: {m['asymmetric_kernel_max_error']:.3g}.\nImpulse maximum error: {m['impulse_max_error']:.3g}.\nExplicit small-example DFT error: {result['example']['explicit_dft_max_abs_error']:.3g}.\nCircular versus zero-boundary maximum error: {m['circular_vs_zero_boundary_max_error']:.6f}.", .760, size=9.8, step=.018)
        put_picture(f, out/'padding_comparison.png', [.055,.465,.89,.20])
        y = paragraph(f, 'Without full padding, the input-size FFT produces circular convolution. Here the kernel is centered consistently, and the circular result is checked against boundary="wrap". The interior still agrees, but wrapped pixels change the edges. This is a boundary-model mismatch, not a failure of the convolution theorem.', .445, size=9.8, width=101, step=.021)
        f.text(.075, y-.005, 'Reproduce from the downloaded repository', fontsize=11, weight='bold', color=BLUE)
        for j,line in enumerate(['python -m pip install -r requirements.txt','python module3.py','python record_demo.py --show']):
            f.text(.085, y-.04-j*.025, line, fontsize=9, family='monospace', color=INK)
        y = paragraph(f, 'The MP4 records actual application output from a fresh execution, with captions and reading holds. Use --show to display the same window. It is not an OS desktop capture and has no audio.\nReview the implementation and results before submission. No script submits to Classroom.', y-.13, size=9.1, width=110, step=.019)
        f.text(.075, y-.002, 'Sources', fontsize=9.5, weight='bold', color=BLUE)
        sources=[('Module 3 assignment',ASSIGNMENT),('SciPy convolve2d','https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html'),
                 ('NumPy fft2 and inverse transform','https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html')]
        for j,(label,url) in enumerate(sources):f.text(.075,y-.025-j*.022,label,fontsize=8.5,color='#166a9b',url=url)
        pdf.savefig(f); plt.close(f)


def save_outputs(result, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'convolution_arrays.npz', **{k:result[k] for k in
                        ['image','kernel','spatial','fourier','difference','image_fft','kernel_fft','product','circular']})
    np.savetxt(out/'gaussian_kernel.csv', result['kernel'], delimiter=',', fmt='%.17g')
    write_json(out/'metrics.json', result['metrics'])
    write_json(out/'tiny_example.json', result['example'])
    write_json(out/'TESTED_ENVIRONMENT.json', {'python':platform.python_version(),'numpy':np.__version__,
              'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'pillow':PIL.__version__})
    make_figures(result,out)
    make_report(result,out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--kernel-size', type=int, default=15)
    parser.add_argument('--sigma', type=float, default=2.5)
    parser.add_argument('--out', type=Path, default=ROOT)
    args = parser.parse_args()
    result = run_experiment(args.image, args.kernel_size, args.sigma)
    save_outputs(result,args.out)
    print(json.dumps(result['metrics'],indent=2))
