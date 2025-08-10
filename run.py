import os
from cloome.model import CLOOME
from huggingface_hub import hf_hub_download

# get checkpoint from Hugging Face
FILENAME = "cloome-bioactivity.pt"
REPO_ID = "anasanchezf/cloome"
ckpt = hf_hub_download(REPO_ID, FILENAME)

config = "src/training/model_configs/RN50.json"
images = [os.path.join("example", "images", f"{channel}.tif") for channel in ["Mito", "ERSyto", "ERSytoBleed", "Ph_golgi", "Hoechst"]]


encoder = CLOOME(ckpt, config)
img_embeddings = encoder.encode_images(images)