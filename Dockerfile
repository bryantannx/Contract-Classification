FROM python:3.7

WORKDIR /app

RUN pip install pandas numpy scikit-learn flask gunicorn

ADD server.py server.py
ADD clf_model.pkl clf_model.pkl
ADD vec.pkl vec.pkl

EXPOSE 3000

CMD [ "gunicorn", "--bind", "0.0.0.0:3000", "server:app" ]