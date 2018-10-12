FROM tensorflow/tensorflow:latest-devel-gpu-py3

RUN mkdir -p /data
VOLUME /data

WORKDIR /data

RUN python -m pip install jupyterlab gputil psutil humanize seaborn keras tables nltk faker tqdm babel gensim && \
    pip install --upgrade pip

EXPOSE 8888

CMD ["jupyter", "lab", "--allow-root", "--no-browser"]
