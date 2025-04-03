import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from onnx import TensorProto
from PIL import Image
from typing import List


def preprocess(image_data: np.ndarray) -> np.ndarray:

    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_image_data = np.zeros(image_data.shape).astype('float32')
    for i in range(image_data.shape[0]):
        # For each pixel in each channel,
        # divide the value by 255 to get value between [0, 1] and then normalize.
        norm_image_data[i, :, :] = (image_data[i, :, :] / 255 -
                                    mean_vec[i]) / stddev_vec[i]
    return norm_image_data


def read_label_data(label_file_path: str) -> List[str]:

    labels = []
    with open(label_file_path, "r") as fhand:
        for line in fhand:
            labels.append(line.strip())

    return labels


def read_test_data(tensor_proto_file_path: str) -> np.ndarray:

    # Option 1: Using TensorProto APIs.
    # tensor = TensorProto()
    # with open(tensor_proto_file_path, "rb") as proto_file:
    #     tensor.ParseFromString(proto_file.read())
    # Option 2: Using ONNX wrapper function instead.
    tensor = onnx.load_tensor(tensor_proto_file_path)
    numpy_array = numpy_helper.to_array(tensor)

    return numpy_array


if __name__ == "__main__":

    onnx_model_file_path = "data/models/resnet18-v1-7.onnx"
    test_input_file_path = "data/test_data/resnet18-v1-7/test_data_set_0/input_0.pb"
    test_output_file_path = "data/test_data/resnet18-v1-7/test_data_set_0/output_0.pb"
    image_file_path = "data/images/european-bee-eater-2115564_1920.jpg"
    label_file_path = "data/labels/synset.txt"

    model_proto = onnx.load(onnx_model_file_path)
    onnx.checker.check_model(model_proto)
    model_proto_bytes = model_proto.SerializeToString()
    # Create ONNX Runtime inference session.
    # https://onnxruntime.ai/docs/get-started/with-python.html
    ort_sess = ort.InferenceSession(
        model_proto_bytes,
        providers=["CPUExecutionProvider", "CUDAExecutionProvider"])

    # Read IO tensor information.
    # https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.NodeArg
    input_nodes = ort_sess.get_inputs()
    input_names = [node.name for node in input_nodes]
    input_shapes = [node.shape for node in input_nodes]
    input_types = [node.type for node in input_nodes]
    output_nodes = ort_sess.get_outputs()
    output_names = [node.name for node in output_nodes]
    output_shapes = [node.shape for node in output_nodes]
    output_types = [node.type for node in output_nodes]

    # Read test input and output files.
    input_array = read_test_data(tensor_proto_file_path=test_input_file_path)
    output_array = read_test_data(tensor_proto_file_path=test_output_file_path)

    # Run unit test.
    output_tensors = ort_sess.run(output_names=output_names,
                                  input_feed={input_names[0]: input_array},
                                  run_options=None)
    assert np.allclose(output_tensors[0], output_array)

    # Read labels.
    labels = read_label_data(label_file_path=label_file_path)

    image_data = np.asarray(Image.open(image_file_path).resize(
        (224, 224))).transpose(2, 0, 1)
    normalized_image_data = preprocess(image_data=image_data)
    normalized_image_data = np.expand_dims(normalized_image_data, axis=0)
    output_tensors = ort_sess.run(
        output_names=output_names,
        input_feed={input_names[0]: normalized_image_data},
        run_options=None)
    predicted_label_idx = np.argmax(output_tensors[0][0])

    print(f"Predicted Label ID: {predicted_label_idx}")
    print(f"Predicted Label: {labels[predicted_label_idx]}")
