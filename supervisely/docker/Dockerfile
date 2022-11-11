FROM supervisely/base-pytorch:6.0.22

RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install supervisely==6.68.17
RUN pip install mmcv==1.3.5
RUN pip install opencv-python-headless==4.5.5.64
RUN pip install opencv-python==4.5.5.64
RUN pip install imagecorruptions==1.1.2