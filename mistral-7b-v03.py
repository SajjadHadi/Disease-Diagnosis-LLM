import os
from datetime import datetime

import modal
import torch
import wandb
from datasets import load_dataset, Dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

BASE_PROJECT_NAME = 'Disease-Diagnosis'
BASE_MODEL = "mistralai/Mistral-7B-v0.3"
PROJECT_NAME = f"{BASE_PROJECT_NAME}-{BASE_MODEL.split('/')[1]}"
HF_USER = "sajjadhadi"

HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_NAME}"
RUN_NAME = f"{datetime.now():%Y-%m-%d}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"

OUTPUT_DIR = "/output"
TOKENIZED_DATA_PATH = f"{OUTPUT_DIR}/tokenized_dataset"

DATASET_NAME = "sajjadhadi/disease-diagnosis-dataset"
RESPONSE_TEMPLATE = " The patient may have"  # PAY ATTENTION TO THE BEGINNING EMPTY SPACE!

# Define the Modal image
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

app = modal.App(PROJECT_NAME, image=image)
volume = modal.Volume.from_name(f'{PROJECT_NAME}-outputs', create_if_missing=True)


# Helper function to tokenize dataset
def tokenize_dataset(dataset, tokenizer, max_seq_length):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )

    return dataset.map(tokenize_function, batched=True, remove_columns=["text"])


@app.function(
    gpu="H100",
    volumes={"/output": volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("llm-fine-tuning-secrets")]
)
def fine_tune(response_template: str, dataset_name: str, is_resume_mode: bool = False):
    # Constants
    LORA_R = 32
    LORA_ALPHA = 64
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    LORA_DROPOUT = 0.1
    EPOCHS = 1
    BATCH_SIZE = 64
    GRADIENT_ACCUMULATION_STEPS = 1
    LEARNING_RATE = 1e-4
    LR_SCHEDULER_TYPE = 'cosine'
    WARMUP_RATIO = 0.03
    OPTIMIZER = "paged_adamw_32bit"
    STEPS = 50
    SAVE_STEPS = 500
    MAX_SEQUENCE_LENGTH = 93

    # Authenticate with Hugging Face
    HF_SECRET = os.getenv("HF_SECRET")
    login(HF_SECRET, add_to_git_credential=True)

    # Weights & Biases setup
    log_to_wandb = os.getenv("WANDB_API_KEY") is not None
    if log_to_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        os.environ["WANDB_PROJECT"] = PROJECT_NAME
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        os.environ["WANDB_WATCH"] = "gradients"
        if is_resume_mode:
            wandb.init(project=PROJECT_NAME, name=RUN_NAME)
        else:
            wandb.init(project=PROJECT_NAME, name=PROJECT_NAME)

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Check if tokenized dataset exists in Volume
    if os.path.exists(TOKENIZED_DATA_PATH):
        print(f"Loading pre-tokenized dataset from {TOKENIZED_DATA_PATH}")
        train_dataset = Dataset.load_from_disk(TOKENIZED_DATA_PATH)
    else:
        print("Tokenizing dataset...")
        dataset = load_dataset(dataset_name)
        train_dataset = dataset['train']
        train_dataset = tokenize_dataset(train_dataset, tokenizer, MAX_SEQUENCE_LENGTH)
        train_dataset.save_to_disk(TOKENIZED_DATA_PATH)
        volume.commit()  # Persist the tokenized dataset
        print(f"Tokenized dataset saved to {TOKENIZED_DATA_PATH}")

    # Quantization config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
    )
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Data collator
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # LoRA configuration
    lora_parameters = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    # Training configuration
    sft_configs = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": 1,
        "eval_strategy": "no",
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "optim": OPTIMIZER,
        "save_steps": SAVE_STEPS,
        "save_total_limit": 10,
        "logging_steps": STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.001,
        "fp16": False,
        "bf16": True,
        "max_grad_norm": 0.3,
        "max_steps": -1,
        "warmup_ratio": WARMUP_RATIO,
        "group_by_length": True,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "report_to": "wandb" if log_to_wandb else None,
        "max_seq_length": MAX_SEQUENCE_LENGTH,
        "dataset_text_field": "text",  # This might need adjustment if tokenized dataset has no 'text' column
        "save_strategy": "steps",
        "hub_strategy": "every_save",
        "push_to_hub": True,
        "hub_model_id": HUB_MODEL_NAME,
        "hub_private_repo": True,
    }

    if is_resume_mode:
        sft_configs["run_name"] = RUN_NAME
        sft_configs["resume_from_checkpoint"] = True

    train_parameters = SFTConfig(**sft_configs)

    # Initialize trainer
    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        peft_config=lora_parameters,
        tokenizer=tokenizer,
        args=train_parameters,
        data_collator=collator
    )

    # Train the model
    if is_resume_mode:
        fine_tuning.train(resume_from_checkpoint=True)
    else:
        fine_tuning.train()

    # Save to Hugging Face Hub
    fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True)
    print(f"Saved to the hub: {PROJECT_RUN_NAME}")
    volume.commit()

    if log_to_wandb:
        wandb.finish()


@app.local_entrypoint()
def main():
    fine_tune.remote(
        response_template=RESPONSE_TEMPLATE,
        dataset_name=DATASET_NAME,
        is_resume_mode=False
    )
