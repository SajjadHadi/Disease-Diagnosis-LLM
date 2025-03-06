from datetime import datetime

base_config = {
    # Base configuration
    'base_project_name': 'Disease-Diagnosis',
    'base_model': "",  # To be initiated during usage
    'hf_user': "sajjadhadi",

    # Derived configurations
    'project_name': "",  # To be initiated during usage
    'hub_model_name': "",  # To be initiated during usage
    'project_run_name': "",  # To be initiated during usage
    'run_name': "",  # To be initiated during usage
    'output_dir': "/output",
    'tokenized_data_path': "/output/tokenized_dataset",

    # Training parameters
    'epochs': 1,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'max_sequence_length': 93,
    'gradient_accumulation_steps': 1,

    # LoRA configuration
    'lora_r': 32,
    'lora_alpha': 64,
    'lora_dropout': 0.1,
    'lora_bias': "none",
    'target_modules': ["q_proj", "v_proj", "k_proj", "o_proj"],

    # Dataset configuration
    'dataset_name': "sajjadhadi/disease-diagnosis-dataset",
    'response_template': " The patient may have",

    # Optimization config
    'optimizer': "paged_adamw_32bit",
    'lr_scheduler_type': 'cosine',
    'warmup_ratio': 0.03,

    # Quantization config
    'quantization_config': {
        'load_in_4bit': True,
        'bnb_4bit_use_double_quant': True,
        'bnb_4bit_compute_dtype': 'bfloat16',
        'bnb_4bit_quant_type': "nf4"
    },

    # Runtime config
    'is_resume_mode': False,
    'steps': 200,
    'save_steps': 500,

    # Modal Config
    'gpu': 'H100',
    'timeout': 7200
}


def get_config(base_model: str, **kwargs):
    base_config['base_model'] = base_model
    base_config['project_name'] = f"{base_config['base_project_name']}-{base_config['base_model'].split('/')[1]}"
    base_config['hub_model_name'] = f"{base_config['hf_user']}-{base_config['project_name']}"
    base_config['run_name'] = f"{datetime.now():%Y-%m-%d_%H.%M}"
    base_config['project_run_name'] = f"{base_config['project_name']}-{base_config['run_name']}"

    for key, val in kwargs.items():
        base_config[key] = val
    return base_config
