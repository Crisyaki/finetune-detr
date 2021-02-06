# Detection Transformers(DETR) finetuning

El conjunto de datos utilizado es un subconjunto de [**Open Images V6**](https://storage.googleapis.com/openimages/web/index.html) que contiene 300 imágenes de pandas rojos y 300 de ratones.


Para descargar las imágenes de Open Images V6 hay que instalar: 
```{bash}
pip install oidv6
``` 
Para descargar las imágenes hay que ejecutar por consola el oidv6, indicando la carpeta donde descargar las imágenes, la clase o clases a descargar y el limite de imágenes. Las instrucciones de uso se encuentran en el Github de [DmitryRyumin](https://github.com/DmitryRyumin/OIDv6). En nuestro caso se ejecutó:
```{bash}
oidv6 downloader en --dataset TransformerDataset\ --type_data train --classes "Red panda" --limit 300 --yes
oidv6 downloader en --dataset TransformerDataset\ --type_data train --classes "Mouse" --limit 300 --yes
```

Las imágenes descargadas contienen las etiquetas en un txt por imagen, por lo que se realizará las trasformaciones necesarias para guardarlas todas en un archivo csv y poder utilizarlas como conjunto de entrenamiento. Ver Notebook  Labels to CSV DataFrame ().

Para esta práctica se estará usando Jupyter notebook desde un entorno de Anaconda, por lo que realizaremos las instalaciones de todo lo necesario por la consola de Anaconda (Anaconda prompt), por ejemplo para instalar la libería extra de pythorch llamada vision: https://github.com/pytorch/vision.
```{bash}
conda install torchvision -c pytorch 
``` 

En este proyecto hace falta instalar:
+ Pandas
+ OpenCV
+ Sklearn
+ Albumentations
+ Pillow (PIL)
+ Pytorch (para CUDA)
+ Torchvision

Antes de proceder a implementar el modelo hay que copiar el repositorio de DETR de Facebook ya que contiene implemetaciones necesarias para la arquitectura como su función de pérdida denonimada **Bipartite Matching loss**, en la que se asigna un bbox de verdad de terreno a un cuadro predicho usando un comparador tanto, cuando se realiza un ajuste fino, necesitamos el comparador (por ejemplo, **hungarian matcher** que se usa en el paper),también necesitamos la función **SetCriterion** que le da a la función de pérdida el backpropogation.

Por lo tanto utilizaremos git, para clonar el repositorio desde github a nuestro equipo utilizando la consola:
```{bash}
git clone https://github.com/facebookresearch/detr.git 
``` 
