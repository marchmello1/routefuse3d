FROM --platform=linux/amd64 pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app
RUN python -m venv --system-site-packages --without-pip /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"
COPY --chown=user:user container/requirements.txt /opt/app/requirements.txt
RUN python -m pip install --no-cache-dir --no-color -r /opt/app/requirements.txt
COPY --chown=user:user container/app.py container/inference.py /opt/app/
COPY --chown=user:user src/model.py /opt/app/model.py
ARG MODEL_SOURCE=model
COPY --chown=user:user ${MODEL_SOURCE} /opt/app/resources
LABEL org.grand-challenge.api-method="invoke"
ENTRYPOINT ["python", "app.py"]
