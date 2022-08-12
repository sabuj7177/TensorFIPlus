import os
import time

from tensorflow.keras.applications import vgg16, vgg19, resnet, xception, nasnet, mobilenet, mobilenet_v2, \
    inception_resnet_v2, inception_v3, densenet
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src import tensorfi_plus as tfi
from src.utility import get_fault_injection_configs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_model_from_name(model_name):
    if model_name == "ResNet50":
        return resnet.ResNet50()
    elif model_name == "ResNet101":
        return resnet.ResNet101()
    elif model_name == "ResNet152":
        return resnet.ResNet152()
    elif model_name == "VGG16":
        return vgg16.VGG16()
    elif model_name == "VGG19":
        return vgg19.VGG19()
    elif model_name == "Xception":
        return xception.Xception()
    elif model_name == "NASNetMobile":
        return nasnet.NASNetMobile()
    elif model_name == "NASNetLarge":
        return nasnet.NASNetLarge()
    elif model_name == "MobileNet":
        return mobilenet.MobileNet()
    elif model_name == "MobileNetV2":
        return mobilenet_v2.MobileNetV2()
    elif model_name == "InceptionResNetV2":
        return inception_resnet_v2.InceptionResNetV2()
    elif model_name == "InceptionV3":
        return inception_v3.InceptionV3()
    elif model_name == "DenseNet121":
        return densenet.DenseNet121()
    elif model_name == "DenseNet169":
        return densenet.DenseNet169()
    elif model_name == "DenseNet201":
        return densenet.DenseNet201()


def get_preprocessed_input_by_model_name(model_name, x_val):
    if model_name == "ResNet50" or model_name == "ResNet101" or model_name == "ResNet152":
        return resnet.preprocess_input(x_val)
    elif model_name == "VGG16":
        return vgg16.preprocess_input(x_val)
    elif model_name == "VGG19":
        return vgg19.preprocess_input(x_val)
    elif model_name == "Xception":
        return xception.preprocess_input(x_val)
    elif model_name == "NASNetMobile" or model_name == "NASNetLarge":
        return nasnet.preprocess_input(x_val)
    elif model_name == "MobileNet":
        return mobilenet.preprocess_input(x_val)
    elif model_name == "MobileNetV2":
        return mobilenet_v2.preprocess_input(x_val)
    elif model_name == "InceptionResNetV2":
        return inception_resnet_v2.preprocess_input(x_val)
    elif model_name == "InceptionV3":
        return inception_v3.preprocess_input(x_val)
    elif model_name == "DenseNet121" or model_name == "DenseNet169" or model_name == "DenseNet201":
        return densenet.preprocess_input(x_val)


def get_input_dim_by_model_name(model_name):
    if model_name == "ResNet50" or model_name == "ResNet101" or model_name == "ResNet152" or model_name == "VGG16" \
            or model_name == "VGG19" or model_name == "NASNetMobile" or model_name == "MobileNet" \
            or model_name == "MobileNetV2" or model_name == "DenseNet121" or model_name == "DenseNet169" \
            or model_name == "DenseNet201" or model_name == "EfficientNetB0":
        return 224
    elif model_name == "Xception" or model_name == "InceptionResNetV2" or model_name == "InceptionV3":
        return 299
    elif model_name == "NASNetLarge":
        return 331


def main():
    model_list = ["ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19", "MobileNet", "MobileNetV2", "DenseNet121",
                  "DenseNet169", "DenseNet201", "Xception", "InceptionResNetV2", "InceptionV3"]
    for model_name in model_list:
        print(model_name)
        model = get_model_from_name(model_name)
        input_dim = get_input_dim_by_model_name(model_name)
        path = 'ILSVRC2012_val_00000001.JPEG'
        image = load_img(path, target_size=(input_dim, input_dim))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = get_preprocessed_input_by_model_name(model_name, image)
        start = time.time()
        out = model.predict(image).argmax(axis=-1)[0]
        end = time.time()
        print("Infer time" + str(end - start))
        print(out)
        model_graph, super_nodes = get_fault_injection_configs(model)
        conf = 'confFiles/sample1.yaml'
        model_layers_len = len(model.layers)
        print("Total layers " + str(model_layers_len))
        for i in range(model_layers_len - 3):
            start = time.time()
            image = load_img(path, target_size=(input_dim, input_dim))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = get_preprocessed_input_by_model_name(model_name, image)
            print("Injecting at " + str(i+1))
            res = tfi.inject(model=model, x_test=image, confFile=conf, model_graph=model_graph, super_nodes=super_nodes,
                             inject_index=i+1)
            done = time.time()
            inject_time = done - start
            print(res.final_label)
            print("inject time " + str(inject_time))


if __name__ == '__main__':
    main()
