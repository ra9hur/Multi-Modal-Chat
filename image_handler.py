from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

import base64

# Base64 images refer to images (binary data) that have been encoded as a Base64 string. 
# This string can be embedded directly into the HTML code of a web page and displayed as an image, 
# without the need for separate image files.
def convert_bytes_to_base64(image_bytes):
    encoded_string = base64.b64encode(image_bytes).decode("utf-8")
    return "data:image/jpeg;base64," + encoded_string


def handle_image(image_bytes, user_message=None):

    # mmproj is a clip model which is fine tuned on text image pairs 
    # this model is going to be used to create image embeddings
    chat_handler = Llava15ChatHandler(clip_model_path="./model/llava/mmproj-model-f16.gguf")

    # llama model fine-tuned to handle the image embeddings
    llm = Llama(model_path="./model/llava/ggml-model-q5_k.gguf",
                chat_handler=chat_handler,
                n_gpu_layers=20,
                logits_all=True,    # needed to make llava work
                n_ctx=1024,         # n_ctx should be increased to accomodate the image embedding size
    )
    
    image_base64 = convert_bytes_to_base64(image_bytes)

    # output as a dictionary
    output = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": "You are an assistant who perfectly describes images."},
            {
                "role": "user",
                "content": [
                    # accepts image urls OR base64 encoded images
                    {"type": "image_url", "image_url": {"url": image_base64}}, 
                    {"type" : "text", "text": user_message}
                ]
            }
        ]
    )

    print(f"User message: %s" % user_message)

    print(output)
    # Returns text output
    return output["choices"][0]["message"]["content"]


def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string=  base64.b64encode(image_file.read()).decode("utf-8")
        return "data:image/jpeg;base64," + encoded_string
    


if __name__ == "__main__":
    image_path = "./Image000.jpg"
    image_base64 = convert_image_to_base64(image_path)
    with open("image.txt", "w") as f:
        f.write(image_base64)

