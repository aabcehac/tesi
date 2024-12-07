FROM jupyter/scipy-notebook

COPY tesi.ipynb .

CMD ["python", "tesi.ipynb"]