#!/usr/bin/env python3
"""Record actual Module 3 computation outputs as a captioned MP4.

Install: python -m pip install -r requirements.txt
Run: python record_demo.py
Display the same working application: python record_demo.py --show

The experiment is rerun, and its progress callbacks provide each displayed
array and numerical result. Frames are held for reading. This records the
application's rendered output, not the operating-system desktop. No audio,
student narration, camera footage or physical experiment is claimed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import textwrap
import time
import cv2
import numpy as np
import module3 as app

ROOT = Path(__file__).resolve().parent
FPS, SIZE = 20, (1280, 720)
BG, INK, BLUE = (248, 251, 253), (58, 46, 31), (80, 62, 24)


def line(canvas, text, x, y, scale=.72, color=INK, bold=False):
    cv2.putText(canvas, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 2 if bold else 1, cv2.LINE_AA)


def paragraph(canvas, text, x, y, chars=92, scale=.72, gap=35):
    for row in textwrap.wrap(text, chars):
        line(canvas,row,x,y,scale); y += gap
    return y


def panel(title):
    frame = np.full((720,1280,3), BG, np.uint8)
    cv2.rectangle(frame,(0,0),(1279,106),BLUE,-1)
    line(frame,'MODULE 3 / '+title,36,48,.95,(255,255,255),True)
    line(frame,'Image blurring: spatial convolution = Fourier multiplication',36,83,.69,(235,229,218))
    cv2.rectangle(frame,(0,668),(1279,719),BLUE,-1)
    line(frame,'Actual computation outputs | Captioned demonstration | Generated test image',28,700,.65,(255,255,255))
    return frame


def put_array(frame, array, box, normalize=False, color=False):
    x,y,w,h = box
    values = np.asarray(array, dtype=float)
    if normalize:
        span = values.max()-values.min()
        values = (values-values.min())/(span if span else 1.)
    pixels = np.uint8(np.clip(values,0,1)*255)
    pixels = cv2.applyColorMap(pixels,cv2.COLORMAP_INFERNO) if color else cv2.cvtColor(pixels,cv2.COLOR_GRAY2BGR)
    factor = min(w/pixels.shape[1],h/pixels.shape[0])
    pixels = cv2.resize(pixels,(round(pixels.shape[1]*factor),round(pixels.shape[0]*factor)),interpolation=cv2.INTER_NEAREST)
    hh,ww = pixels.shape[:2]
    left,top = x+(w-ww)//2,y+(h-hh)//2
    frame[top:top+hh,left:left+ww] = pixels


def main(show=False):
    saved = json.loads((ROOT/'metrics.json').read_text())
    writer = cv2.VideoWriter(str(ROOT/'Module_3_Working_Demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),FPS,SIZE)
    if not writer.isOpened(): raise RuntimeError('MP4 encoder unavailable')
    events, frames = [], 0
    started = time.perf_counter()

    def capture(frame, seconds, event):
        nonlocal frames
        count = round(seconds*FPS)
        events.append({'stage':event,'first_frame':frames,'frames':count,
                       'execution_elapsed_seconds':time.perf_counter()-started})
        for _ in range(count):
            current = frame.copy()
            line(current,f'Playback {frames//FPS} s',1100,643,.48)
            writer.write(current); frames += 1
            if show:
                cv2.imshow('Module 3 - spatial and Fourier blurring',current)
                if cv2.waitKey(round(1000/FPS))&255 == 27: raise KeyboardInterrupt

    def progress(stage,data):
        if stage == 'input':
            f = panel('Input image and Gaussian kernel')
            line(f,'192 x 256 grayscale test pattern',45,151,.76,BLUE,True)
            put_array(f,data['image'],(40,175,670,410))
            line(f,'15 x 15 Gaussian',800,153,.77,BLUE,True)
            put_array(f,data['kernel'],(835,190,235,235),normalize=True)
            paragraph(f,'sigma = 2.5 pixels; sum = 1. Kernel display contrast is normalized.',760,473,39,.70,33)
            line(f,'Deterministic mathematical image; values on [0,1], stored as float64.',45,625,.73)
            capture(f,7,stage)
        elif stage == 'spatial':
            f = panel('Spatial convolution computed')
            line(f,'Original image',100,154,.8,BLUE,True)
            line(f,'Direct spatial Gaussian blur',710,154,.8,BLUE,True)
            put_array(f,data['image'],(50,180,560,350))
            put_array(f,data['output'],(665,180,560,350))
            paragraph(f,'Each output pixel is a weighted sum of nearby input intensities. Zero extension outside the input gives a full 206 x 270 result. These views use the same centered 192 x 256 crop.',50,577,100,.72,32)
            capture(f,7,stage)
        elif stage == 'frequency':
            f = panel('FFT, pointwise multiplication, inverse FFT')
            arrays=[np.log1p(np.abs(np.fft.fftshift(data['image_fft']))),
                    np.abs(np.fft.fftshift(data['kernel_fft'])),data['output']]
            labels=['log(1 + |F|)','Gaussian |H|','Real IFFT(F x H)']
            for i,(a,label) in enumerate(zip(arrays,labels)):
                line(f,label,55+i*414,160,.76,BLUE,True)
                put_array(f,a,(28+i*414,190,390,325),normalize=i<2)
            paragraph(f,'Pad both arrays to 206 x 270 before transforming. Multiply the complex coefficients element by element; then inverse transform. fftshift is used only to display spectra.',50,562,100,.73,35)
            capture(f,8,stage)
        elif stage == 'comparison':
            f = panel('Both methods agree to numerical round-off')
            arrays=[data['spatial'],data['fourier'],data['difference']]
            labels=['Spatial blur','Fourier blur','Difference, magnified']
            for i,(a,label) in enumerate(zip(arrays,labels)):
                line(f,label,45+i*414,156,.73,BLUE,True)
                put_array(f,a,(28+i*414,180,390,315),normalize=i==2,color=i==2)
            line(f,f"Full-output maximum absolute error: {data['max_error']:.9e}",50,547,.88,BLUE,True)
            line(f,f"Full-output root mean squared error: {data['rmse']:.9e}",50,592,.86)
            line(f,f"Difference colors span 0 to {data['difference'].max():.3e}; they are not visible image errors.",50,631,.66)
            capture(f,8,stage)
        elif stage == 'padding':
            f = panel('Why zero-padding matters')
            for i,(a,label) in enumerate(zip([data['zero_boundary'],data['circular'],data['difference']],
                                             ['Zero boundary','Circular boundary','Boundary difference'])):
                line(f,label,45+i*414,156,.75,BLUE,True)
                put_array(f,a,(28+i*414,180,390,315))
            line(f,f"Circular versus zero-boundary maximum difference: {data['max_error']:.6f}",50,546,.83,BLUE,True)
            paragraph(f,'An input-size FFT wraps opposite edges. The consistently centered circular result matches SciPy boundary="wrap". Interior pixels still agree with zero padding; the edge model changed.',50,588,103,.68,30)
            capture(f,8,stage)
        elif stage == 'example':
            f = panel('Independent, hand-checkable example')
            line(f,'2 x 2 image',65,177,.86,BLUE,True)
            line(f,'Averaging kernel',430,177,.86,BLUE,True)
            line(f,'Full convolution',850,177,.86,BLUE,True)
            for x,matrix in [(65,data['image']),(430,data['kernel']),(850,data['expected'])]:
                for j,row in enumerate(matrix):
                    line(f,'[ '+'   '.join(f'{v:g}' for v in row)+' ]',x,255+j*62,1.0)
            line(f,'Center = (1 + 2 + 3 + 4) / 4 = 2.5',65,483,1.,BLUE,True)
            paragraph(f,'Explicit 3 x 3 DFT matrices reproduce this result independently of the FFT implementation. This directly checks convolution in space against multiplication in frequency.',65,544,92,.76,35)
            capture(f,8,stage)
        elif stage == 'validation':
            f = panel('Eight numerical checks passed')
            labels=['Full spatial / FFT agreement','Centered crop matches same mode','Asymmetric kernel orientation','Impulse origin and alignment',
                    'Full output sum conserved','Explicit DFT worked example','Circular result matches wrap','Interior boundary agreement']
            for i,label in enumerate(labels):
                x,y = 55+(i//4)*620,202+(i%4)*84
                line(f,'PASS  '+label,x,y,.69,BLUE,True)
            paragraph(f,'The code uses assertions before marking checks as passed. Actual arrays, metrics, the exact kernel, environment versions and report are included with the source.',55,575,99,.71,32)
            assert data['all_checks_passed']
            for key in ['max_abs_error','rmse','circular_vs_zero_boundary_max_error']:
                np.testing.assert_allclose(data[key],saved[key],rtol=1e-12,atol=1e-15)
            capture(f,8,stage)

    try:
        f = panel('Working application demonstration')
        line(f,'Convolution in space = multiplication in frequency',50,190,1.,BLUE,True)
        for j,row in enumerate(['1. Generate the image and normalized Gaussian kernel.',
                                '2. Execute spatial convolution and padded FFT filtering.',
                                '3. Compare numerical results and inspect boundary effects.',
                                '4. Verify a small example with an independent DFT.']):
            line(f,row,55,285+j*66,.78)
        paragraph(f,'This recording captures actual computation outputs with reading holds. It is an application-output recording, without audio or an OS desktop capture.',55,576,97,.70,32)
        capture(f,6,'start')
        result = app.run_experiment(progress=progress)
        f = panel('Reproduce and inspect the results')
        for j,row in enumerate(['python -m pip install -r requirements.txt','python module3.py','python record_demo.py --show']):
            line(f,row,60,220+j*90,1.,BLUE,True)
        paragraph(f,'The first run regenerates the report, arrays and figures. The second reruns the experiment and displays the same working application while recording its outputs.',60,505,94,.77,36)
        line(f,'Source and calculations are included for review.',60,628,.65)
        capture(f,6,'finish')
    finally:
        writer.release()
        if show: cv2.destroyAllWindows()

    video = cv2.VideoCapture(str(ROOT/'Module_3_Working_Demo.mp4'))
    decoded, dimensions = 0, None
    while True:
        ok,frame = video.read()
        if not ok: break
        dimensions = [frame.shape[1],frame.shape[0]]
        assert dimensions == list(SIZE)
        decoded += 1
        if decoded == 800: cv2.imwrite('/tmp/module3-demo-comparison.jpg',frame)
    actual_fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    assert decoded == frames and abs(actual_fps-FPS)<1e-6
    app.write_json(ROOT/'DEMO_RECORDING_VALIDATION.json',{
        'recording_type':'Actual application-rendered outputs from a fresh computation; reading holds added.',
        'desktop_capture':False,'audio':False,'fps':FPS,'frame_count':frames,
        'decoded_frames':decoded,'size_pixels':list(SIZE),'duration_seconds':frames/FPS,
        'stages':events,'opencv_version':cv2.__version__,'all_checks_passed':result['metrics']['all_checks_passed'],
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['module3.py','record_demo.py']}})
    print(json.dumps({'video':str(ROOT/'Module_3_Working_Demo.mp4'),'duration_seconds':frames/FPS,'decoded_frames':decoded}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show',action='store_true')
    main(parser.parse_args().show)
