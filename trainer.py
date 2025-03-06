import os

import modal
import torch
import wandb
from datasets import load_dataset, Dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


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


def create_image():
    return (
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


def train_model(config, volume=None):
    # Authenticate with Hugging Face
    HF_SECRET = os.getenv("HF_SECRET")
    login(HF_SECRET, add_to_git_credential=True)

    # Weights & Biases setup
    log_to_wandb = os.getenv("WANDB_API_KEY") is not None
    if log_to_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        os.environ["WANDB_PROJECT"] = config['project_name']
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        os.environ["WANDB_WATCH"] = "gradients"
        if config['is_resume_mode']:
            wandb.init(project=config['project_name'], name=config['run_name'])
        else:
            wandb.init(project=config['project_name'], name=config['project_run_name'])

    tokenizer = AutoTokenizer.from_pretrained(config['base_model'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if os.path.exists(config['tokenized_data_path']):
        print(f"Loading pre-tokenized dataset from {config['tokenized_data_path']}")
        train_dataset = Dataset.load_from_disk(config['tokenized_data_path'])
    else:
        print("Tokenizing dataset...")
        dataset = load_dataset(config['dataset_name'])
        train_dataset = dataset['train']
        train_dataset = tokenize_dataset(train_dataset, tokenizer, config['max_sequence_length'])
        train_dataset.save_to_disk(config['tokenized_data_path'])
        if volume:
            volume.commit()

    quant_config = BitsAndBytesConfig(
        load_in_4bit=config['quantization_config']['load_in_4bit'],
        bnb_4bit_use_double_quant=config['quantization_config']['bnb_4bit_use_double_quant'],
        bnb_4bit_compute_dtype=getattr(torch, config['quantization_config']['bnb_4bit_compute_dtype']),
        bnb_4bit_quant_type=config['quantization_config']['bnb_4bit_quant_type']
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config['base_model'],
        quantization_config=quant_config,
        device_map="auto",
    )
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    collator = DataCollatorForCompletionOnlyLM(config['response_template'], tokenizer=tokenizer)

    lora_parameters = LoraConfig(
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        r=config['lora_r'],
        bias=config['lora_bias'],
        task_type="CAUSAL_LM",
        target_modules=config['target_modules'],
    )

    sft_config = SFTConfig(
        output_dir=config['output_dir'],
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        optim=config['optimizer'],
        save_steps=config['save_steps'],
        logging_steps=config['steps'],
        learning_rate=config['learning_rate'],
        weight_decay=0.001,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=config['warmup_ratio'],
        lr_scheduler_type=config['lr_scheduler_type'],
        report_to="wandb" if log_to_wandb else None,
        max_seq_length=config['max_sequence_length'],
        hub_model_id=config['hub_model_name'],
        push_to_hub=True,
        hub_private_repo=True,
    )

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        peft_config=lora_parameters,
        tokenizer=tokenizer,
        args=sft_config,
        data_collator=collator
    )

    trainer.train(resume_from_checkpoint=config['is_resume_mode'])

    trainer.model.push_to_hub(config['project_run_name'], private=True)
    if log_to_wandb:
        wandb.finish()
