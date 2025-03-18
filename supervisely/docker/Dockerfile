FROM supervisely/base-pytorch:6.0.22

RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install opencv-python-headless==4.5.5.64
RUN pip install opencv-python==4.5.5.64
RUN pip install imgaug==0.4.0 imagecorruptions==1.1.2

RUN pip install mmcv==2.2.0\
    mmengine==0.10.7

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
