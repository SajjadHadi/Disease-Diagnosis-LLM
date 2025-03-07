import modal

from config import get_config
from trainer import train_model

config = get_config(base_model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "peft",
        "bitsandbytes",
        "wandb",
        "accelerate",
        "huggingface_hub",
        "trl",
        "sentencepiece"
    )
    .apt_install("git")
)

app = modal.App(config['project_name'], image=image)
volume = modal.Volume.from_name(f'{config["project_name"]}-outputs', create_if_missing=True)


@app.function(
    gpu=config['gpu'],
    volumes={"/output": volume},
    timeout=config['timeout'],
    secrets=[modal.Secret.from_name("llm-fine-tuning-secrets")]
)
def run_training():
    train_model(config, volume=modal.Volume.lookup(f"{config['project_name']}-outputs"))


@app.local_entrypoint()
def main():
    run_training.remote()