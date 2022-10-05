import numpy as np
import onnxruntime as rt
import PIL.Image as Image

def process_image(img_path, input_shape):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(input_shape)
    image = np.array(img, dtype=np.float32)
    #print(image.shape, image.dtype)
    image = image.transpose((2,0,1))[np.newaxis, ...]
    return image


def main() -> None:

    model_file_path = "deploy.onnx"
    img_path = "input_image.jpg"

    sess = rt.InferenceSession(model_file_path)

    input_dimension = sess.get_inputs()[0].shape
    print("Inputs Dimension:", input_dimension)

    inname = [input.name for input in sess.get_inputs()]
    outname = [output.name for output in sess.get_outputs()]

    input_shape = [input_dimension[3], input_dimension[2]]
    data_input = process_image(img_path, input_shape)

    print("inputs name:",inname,"|| outputs name:",outname)

    prediction = sess.run(outname, {inname[0]: data_input})

    print("session run finished:")
    for i in range(len(prediction)):
        print("Name:", outname[i], ", Dimension:", prediction[i].shape)
        print(prediction[i])

if __name__ == "__main__":

    main()
