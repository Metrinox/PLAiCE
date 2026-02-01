import torch
from diffusers import DiffusionPipeline

# switch to "mps" for apple devices
pipe = DiffusionPipeline.from_pretrained("amused/amused-256", dtype=torch.bfloat16, device_map="cuda")

prompt = "bagel gets no signal in imperial college"
image = pipe(prompt).images[0]

image.save("astronaut_jungle.png")