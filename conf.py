import os
import wget

#
# DIRETORIAS QUE CONSTITUEM A API TFDO.
#

IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')
LABELIMG_PATH = os.path.join('Tensorflow', 'labelimg')

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}


#
# CRIAÇÃO DAS DIRETORIAS QUE CONSTITUEM A API TFDO.
#

for path in paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            os.system("mkdir -p {}".format(path))
        if os.name == 'nt':
            os.system("mkdir {}".format(path))


#
# CRIAÇÃO DAS DIRETORIAS ONDE SÃO GUARDADAS AS IMAGENS DE TREINO E TESTE.
#

labels = ['cocacola', 'pepsi']

if not os.path.exists(IMAGES_PATH):
    if os.name == 'posix':
        os.system("mkdir -p {}".format(IMAGES_PATH))
    if os.name == 'nt':
         os.system("mkdir {}".format(IMAGES_PATH))

for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.system("mkdir {}".format(path))


#
# DOWNLOAD E INSTALAÇÃO DA API TFDO.
#

if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    os.system("git clone https://github.com/tensorflow/models {}".format(paths['APIMODEL_PATH']))

url="https://github.com/protocolbuffers/protobuf/releases/download/v3.19.3/protoc-3.19.3-win64.zip"
wget.download(url)
os.system("move protoc-3.19.3-win64.zip {}".format(paths['PROTOC_PATH']))
os.system("cd {} && tar -xf protoc-3.19.3-win64.zip".format(paths['PROTOC_PATH']))
os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))
os.system("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install")
os.system("cd Tensorflow/models/research/slim && pip install -e .")


#
# SCRIPT QUE VERIFICA SE A INSTALAÇÃO DA API TFDO FOI BEM SUCEDIDA.
#

VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
os.system("python {}".format(VERIFICATION_SCRIPT))


#
# DOWNLOAD DO MODELO QUE VAI SER UTILIZADO.
#

if os.name =='posix':
    os.system("wget {}".format(PRETRAINED_MODEL_URL))
    os.system("mv {}.tar.gz {}".format(PRETRAINED_MODEL_NAME, paths['PRETRAINED_MODEL_PATH']))
    os.system("cd {} && tar -zxvf {}.tar.gz".format(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME))
if os.name == 'nt':
    wget.download(PRETRAINED_MODEL_URL)
    os.system("move {}.tar.gz {}".format(PRETRAINED_MODEL_NAME, paths['PRETRAINED_MODEL_PATH']))
    os.system("cd {} && tar -zxvf {}.tar.gz".format(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME))

#
# COPIA-SE O MODELO PRE TREINADO PARA UMA OUTRA DIRETORIA.
#

if os.name =='posix':
    os.system("cp {} {}".format(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config'),
                                os.path.join(paths['CHECKPOINT_PATH'])))
if os.name == 'nt':
    os.system("copy {} {}".format(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config'),
                                  os.path.join(paths['CHECKPOINT_PATH'])))