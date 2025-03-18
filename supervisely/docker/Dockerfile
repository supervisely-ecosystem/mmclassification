FROM supervisely/base-pytorch:6.0.22

RUN pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url https://download.pytorch.org/whl/cu118

RUN pip install -U openmim
RUN pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
RUN pip install mmengine

RUN pip install opencv-python-headless==4.5.5.64
RUN pip install opencv-python==4.5.5.64
RUN pip install imagecorruptions==1.1.2

RUN pip install urllib3>=1.26.15
RUN pip install requests-toolbelt==1.0.0
RUN pip install yapf==0.40.1
RUN pip install cachetools==4.2.4

RUN pip install supervisely==6.73.158