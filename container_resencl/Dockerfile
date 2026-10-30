FROM --platform=linux/amd64 pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

ARG SELMA_MIRRORING=0
ARG SELMA_MIRROR_AXES=all
ARG SELMA_ARRAY_API=0
ARG SELMA_MIN_COMPONENT_SIZE=1
ARG SELMA_ENSEMBLE_MEMBERS=all
ARG SELMA_ENSEMBLE_FOLDS=all
ENV PYTHONUNBUFFERED=1 nnUNet_compile=false \
    nnUNet_raw=/tmp/nnUNet_raw nnUNet_preprocessed=/tmp/nnUNet_preprocessed \
    nnUNet_results=/opt/app/resources \
    SELMA_MIRRORING=${SELMA_MIRRORING} SELMA_MIRROR_AXES=${SELMA_MIRROR_AXES} \
    SELMA_ARRAY_API=${SELMA_ARRAY_API} \
    SELMA_MIN_COMPONENT_SIZE=${SELMA_MIN_COMPONENT_SIZE} \
    SELMA_ENSEMBLE_MEMBERS=${SELMA_ENSEMBLE_MEMBERS} \
    SELMA_ENSEMBLE_FOLDS=${SELMA_ENSEMBLE_FOLDS}
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app
RUN python -m venv --system-site-packages --without-pip /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"
COPY --chown=user:user container_resencl/requirements.txt /opt/app/requirements.txt
RUN python -m pip install --no-cache-dir --no-color -r /opt/app/requirements.txt
COPY --chown=user:user container/app.py container_resencl/inference.py /opt/app/
ARG MODEL_SOURCE=model
COPY --chown=user:user ${MODEL_SOURCE} /opt/app/resources
LABEL org.grand-challenge.api-method="invoke"
ENTRYPOINT ["python", "app.py"]
