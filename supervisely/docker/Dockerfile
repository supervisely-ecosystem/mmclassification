FROM supervisely/base-pytorch:6.0.22

RUN pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu116
RUN pip install opencv-python-headless==4.8.1.78
RUN pip install opencv-python==4.8.1.78
RUN pip install imgaug==0.4.0 imagecorruptions==1.1.2

RUN pip install --upgrade pip && \
    pip install -U openmim && \
    mim install mmengine==0.7.1 && \
    mim install mmcv-full==2.0.0 && \
    mim install mmpretrain==1.0.0

RUN pip install urllib3>=1.26.15 \
    requests-toolbelt==1.0.0 \
    yapf==0.40.1 \
    cachetools==4.2.4

RUN pip install importlib_metadata==8.6.1 \
    modelindex==0.0.2 \
    einops==0.8.1 \
    mat4py==0.6.0 \
    matplotlib==3.10.1 \
    rich==13.9.4

RUN pip install supervisely==6.73.332
