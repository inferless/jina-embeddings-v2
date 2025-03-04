import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer, util

class InferlessPythonModel:
    def initialize(self):
        model_id = "jinaai/jina-embeddings-v2-base-en"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.model = SentenceTransformer(model_id,trust_remote_code=True)
        # control your input sequence length up to 8192
        self.model.max_seq_length = 1024

    def infer(self, inputs):
        sentences = inputs["sentences"]
        embeddings = self.model.encode(sentences)
        return {"result": embeddings}
    def finalize(self, args):
        self.pipe = None
