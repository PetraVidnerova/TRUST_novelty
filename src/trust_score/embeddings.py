import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from accelerate import dispatch_model

class Embeddings():

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
        self.model = AutoAdapterModel.from_pretrained('allenai/specter2_base', device_map="auto")
        self.model.load_adapter("allenai/specter2",
                                source="hf",
                                load_as="specter2",
                                set_active=True
                                )
                                #device_map="auto") # device map does not work here
        #self.model.to("cuda")

        # this shows some params are still on cpu 
        # for name, param in self.model.named_parameters():
        #     if param.device.type == "cpu":
        #         print("CPU param:", name)
        # exit()

        # so we force them to move on appropriate gpus
        device_map = self.model.hf_device_map
        self.model = dispatch_model(self.model, device_map=device_map)
        self.model.set_active_adapters("specter2")

        print("Active adapters:", self.model.active_adapters)
        
    def embed(self, titles_abstracts, titles_only=False):
        if titles_only:
            text_batch = [
                title
                for title, abstract in titles_abstracts
            ]
        else:
            text_batch = [
                title + self.tokenizer.sep_token + abstract
                for title, abstract in titles_abstracts
            ]
        inputs = self.tokenizer(text_batch, padding=True, truncation=True,
                                return_tensors="pt", return_token_type_ids=False, max_length=512)
       
        device = next(self.model.parameters()).device # which gpu to move the inputs to   
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs)
        embedding = output.last_hidden_state[:, 0, :]
        return embedding
