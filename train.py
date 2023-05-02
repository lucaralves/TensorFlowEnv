import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

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
# CRIAÇÃO DE UM LABELMAP. NECESSÁRIO PARA SE GERAR OS TFRECORDS.
#

labels = [{'name':'CocaCola', 'id':1}, {'name':'Pepsi', 'id':2}]

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')


#
# DOWNLOAD DO SCRIPT QUE GERA OS TFRECORDS. GERAM-SE OS TFRECORDS DAS IMAGENS DE TREINO E TESTE.
#

if not os.path.exists(files['TF_RECORD_SCRIPT']):
    os.system("git clone https://github.com/nicknochnack/GenerateTFRecord {}".format(paths['SCRIPTS_PATH']))

os.system("python {} -x {} -l {} -o {}".format(files['TF_RECORD_SCRIPT'], os.path.join(paths['IMAGE_PATH'], 'train'),
                                               files['LABELMAP'], os.path.join(paths['ANNOTATION_PATH'], 'train.record')))
os.system("python {} -x {} -l {} -o {}".format(files['TF_RECORD_SCRIPT'], os.path.join(paths['IMAGE_PATH'], 'test'),
                                               files['LABELMAP'], os.path.join(paths['ANNOTATION_PATH'], 'test.record')))


#
# CONFIGURA-SE A CÓPIA DO MODELO EM FUNÇÃO DOS NOSSOS DADOS E CLASSES A DISTINGUIR.
#

config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
    f.write(config_text)


#
# TREINA-SE O MODELO E VISUALIZA-SE A SUA PERFORMANCE ATRAVÉS DO TENSORBOARD.
#

TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')

command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=10000".\
    format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'])

os.system(command)

os.system("cd C:\\Users\\TECRA\\Desktop\\Projetos\\TensorFlowEnv\\Tensorflow\\workspace\\models\\my_ssd_mobnet\\train" +
          " && tensorboard --logdir=.")


#
# AVALIAÇÃO GERAL DO MODELO COM AS IMAGENS DE TESTE. VISUALIZAÇÃO DO TESTE ATRAVÉS DO TENSORBOARD.
#

command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])
os.system(command)

os.system("cd C:\\Users\\TECRA\\Desktop\\Projetos\\TensorFlowEnv\\Tensorflow\\workspace\\models\\my_ssd_mobnet\\eval" +
          " && tensorboard --logdir=.")