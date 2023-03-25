FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-dev git vim


ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
ENV FORCE_CUDA="1"

RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html


WORKDIR /workspace/
COPY ./ /workspace
RUN pip install -r /workspace/requirements.txt
RUN pip install numpy --upgrade
RUN pip install pillow --upgrade
RUN python setup.py build develop

# RUN rm -rf AdelaiDet
RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /tmp/*
RUN rm -rf ~/.cache
RUN apt clean all
RUN conda clean -y -a

ENV NVIDIA_VISIBLE_DEVICES all	
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
EXPOSE 8888
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

CMD ["python", "/workspace/main.py"]
