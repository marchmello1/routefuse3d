FROM --platform=linux/amd64 pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 nnUNet_compile=false \
    nnUNet_raw=/tmp/nnUNet_raw nnUNet_preprocessed=/tmp/nnUNet_preprocessed \
    nnUNet_results=/opt/app/specialist \
    SELMA_FALLBACK_RESOURCE_DIR=/opt/app/fallback \
    SELMA_SPECIALIST_RESOURCE_DIR=/opt/app/specialist \
    SELMA_ARRAY_API=1 \
    SELMA_MAX_SPECIALIST_VOXELS=27000000
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app
RUN python -m venv --system-site-packages --without-pip /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"
COPY --chown=user:user container_h200_hybrid/requirements.txt /opt/app/requirements.txt
RUN python -m pip install --no-cache-dir --no-color -r /opt/app/requirements.txt
COPY --chown=user:user container/app.py /opt/app/app.py
COPY --chown=user:user container_h200_hybrid/inference.py /opt/app/inference.py
COPY --chown=user:user container/inference.py /opt/app/fallback_inference.py
COPY --chown=user:user container_resencl/inference.py /opt/app/specialist_inference.py
COPY --chown=user:user src/model.py /opt/app/model.py
COPY --chown=user:user src/hybrid_policy.py /opt/app/hybrid_policy.py
ARG FALLBACK_MODEL_SOURCE=model_fallback
ARG SPECIALIST_MODEL_SOURCE=model_specialist
COPY --chown=user:user ${FALLBACK_MODEL_SOURCE} /opt/app/fallback
COPY --chown=user:user ${SPECIALIST_MODEL_SOURCE} /opt/app/specialist
LABEL org.grand-challenge.api-method="invoke"
ENTRYPOINT ["python", "app.py"]
