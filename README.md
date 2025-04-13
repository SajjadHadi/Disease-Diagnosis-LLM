### Project Introduction: Disease Diagnosis Model Fine-Tuning

This project fine-tunes a transformer-based language model for disease diagnosis using a custom dataset. Leveraging Hugging Face's ecosystem, LoRA for efficient adaptation, and Modal for scalable deployment, it aims to predict potential diagnoses from patient data. The codebase is designed for modularity, reproducibility, and integration with modern ML workflows.

**Features:**
- **Model Fine-Tuning**: Fine-tunes models like DeepSeek-R1-Distill-Llama-8B with LoRA for parameter efficiency.
- **Quantization**: Uses 4-bit quantization (BitsAndBytes) for memory-efficient training.
- **Dataset Handling**: Tokenizes and caches datasets for faster iteration.
- **Experiment Tracking**: Integrates Weights & Biases (W&B) for logging and visualization.
- **Scalable Deployment**: Runs on Modal with H100 GPU support for high-performance training.
- **Hugging Face Integration**: Pushes models to the Hugging Face Hub for sharing and deployment.
- **Resume Capability**: Supports resuming training from checkpoints.

**Best Practices:**
- **Modular Configuration**: Centralize settings in `config.py` for easy experimentation.
- **Environment Management**: Use `.env` for sensitive keys (e.g., HF_SECRET, WANDB_API_KEY).
- **Reproducibility**: Cache tokenized datasets and log experiments with W&B.
- **Resource Efficiency**: Apply LoRA and quantization to reduce compute requirements.
- **Version Control**: Commit changes to volumes in Modal for persistence.
- **Error Handling**: Validate environment variables and paths before training.

---

### Guidance to Run the Project

This guide assumes you have a `.env` file with `HF_SECRET` and `WANDB_API_KEY`, and an `environment.yaml` file for local setup.

#### Prerequisites
- **Python**: Version 3.10 or higher.
- **Conda**: For managing the environment.
- **Modal Account**: Sign up at [modal.com](https://modal.com) and configure the CLI.
- **Hugging Face Account**: Ensure you have a token (`HF_SECRET`) with write access.
- **Weights & Biases**: Optional, for experiment tracking (`WANDB_API_KEY`).
- **GPU Access**: Modal runs on an H100 GPU by default (configurable in `config.py`).

#### Steps to Run

1. **Set Up the Environment**
   - Install Conda if not already installed.
   - Create and activate the environment from `environment.yaml`:
     ```bash
     conda env create -f environment.yaml
     conda activate <env_name>
     ```
   - Alternatively, install dependencies manually:
     ```bash
     pip install torch transformers datasets peft bitsandbytes wandb accelerate huggingface_hub trl sentencepiece
     ```

2. **Configure Environment Variables**
   - Ensure your `.env` file contains:
     ```plaintext
     HF_SECRET=<your_huggingface_token>
     WANDB_API_KEY=<your_wandb_api_key>
     ```
   - Load the `.env` file (e.g., using `python-dotenv` or manually exporting):
     ```bash
     export HF_SECRET=<your_huggingface_token>
     export WANDB_API_KEY=<your_wandb_api_key>
     ```

3. **Install Modal CLI**
   - Install the Modal client:
     ```bash
     pip install modal
     ```
   - Authenticate with Modal:
     ```bash
     modal token new
     ```

4. **Create Modal Secret**
   - Create a Modal secret named `llm-fine-tuning-secrets` with your environment variables:
     ```bash
     modal secret create llm-fine-tuning-secrets HF_SECRET=<your_huggingface_token> WANDB_API_KEY=<your_wandb_api_key>
     ```

5. **Run the Training**
   - Execute the training script via Modal:
     ```bash
     modal run run.py
     ```
   - This will:
     - Build the Modal image with dependencies.
     - Run the training job on an H100 GPU.
     - Save outputs to a Modal volume (`<project_name>-outputs`).
     - Push the fine-tuned model to the Hugging Face Hub.

6. **Monitor Progress**
   - If W&B is enabled, check your W&B dashboard for training metrics.
   - Outputs (checkpoints, tokenized datasets) are stored in `/output` (persisted in the Modal volume).

7. **Optional: Local Testing**
   - To test locally (not recommended for large models):
     - Modify `run.py` to call `train_model(config)` directly instead of `run_training.remote()`.
     - Ensure you have sufficient GPU memory (e.g., for 4-bit quantization).

#### Troubleshooting
- **Modal Errors**: Verify your Modal token and secret configuration.
- **Hugging Face Login**: Ensure `HF_SECRET` has write access to your Hub namespace.
- **W&B Issues**: Confirm `WANDB_API_KEY` is valid or disable W&B in `trainer.py` by setting `report_to=None`.
- **Out of Memory**: Adjust `batch_size` or `gradient_accumulation_steps` in `config.py`.

#### Customization
- Change the base model in `run.py` (e.g., `get_config(base_model="another/model")`).
- Update hyperparameters in `config.py` (e.g., `epochs`, `learning_rate`).
- Modify `dataset_name` in `config.py` to use a different dataset.

#### Outputs
- **Model**: Fine-tuned model pushed to `sajjadhadi/<project_name>-<run_name>` on Hugging Face Hub.
- **Dataset**: Tokenized dataset saved to `/output/tokenized_dataset`.
- **Logs**: Training metrics in W&B (if enabled) and local console.

For further details, refer to the codebase or contact the project maintainer.
