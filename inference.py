import  os
from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, TextStreamer, \
    Qwen3VLForConditionalGeneration
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

class MyModel:
    def __init__(self):

        self.MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
        self.ADAPTER_DIR = os.getenv('ADAPTER_PATH')
        self.MAX_LENGTH = 2048
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.MODEL_ID,
            dtype="auto",
            device_map="auto"

        )
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 1280 * 28 * 28
        self.model = PeftModel.from_pretrained(self.model, self.ADAPTER_DIR)

        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, min_pixels=self.min_pixels,
                                                       max_pixels=self.max_pixels)

        self.SYSTEM_PROMPT = 'You are a LaTeX helper.'
        self.USER_PROMPT = '''Write the correct LaTeX expression for the formula on the image.'''

    def infenence(self, image, system_prompt=None, user_prompt=None):

        if system_prompt is None:
            system_prompt = self.SYSTEM_PROMPT

        if user_prompt is None:
            user_prompt = self.USER_PROMPT

        messages = [
            {
                "role": "system",
                "content":
                    [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True,
                                                    return_tensors="pt")

        generated_ids = self.model.generate(**inputs, max_new_tokens=100,
                                            streamer=TextStreamer(self.processor.tokenizer, skip_prompt=True,
                                                                  skip_special_tokens=True))

        return self.processor.tokenizer.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:],
                                                     skip_special_tokens=True)[0]
