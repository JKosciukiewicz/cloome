import os
from cloome.model import CLOOME
from huggingface_hub import hf_hub_download
import torch


# get checkpoint from Hugging Face
FILENAME = "cloome-bioactivity.pt"
REPO_ID = "anasanchezf/cloome"
ckpt = hf_hub_download(REPO_ID, FILENAME)

config = "src/training/model_configs/RN50.json"
#images = [os.path.join("example", "images", f"{channel}.tif") for channel in ["Mito", "ERSyto", "ERSytoBleed", "Ph_golgi", "Hoechst"]]
image_df = pd.read_csv("/net/tscratch/people/plgjkosciukiewi/bbbc021/bbbc_image_paths_2.csv")
images = [os.path.join("example", "images", fname) for fname in image_df['Image_Name']]

encoder = CLOOME(ckpt, config)
img_embeddings = encoder.encode_images(images)

save_path = "/net/pr2/projects/plgrid/plggicv/jk/datasets/bbbc_cloome/img_embeddings.pt"
torch.save(img_embeddings, save_path)
print(f"Embeddings saved to {save_path}")
