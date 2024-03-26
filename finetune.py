import os
from dataclasses import dataclass, field
from typing import Optional
from train_datasets import CustomDataset
from torch.utils.data import DataLoader

import torch
from datasets import load_dataset
from datasets import load_from_disk
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from tqdm.notebook import tqdm

# from huggingface_hub import login, interpreter_login
# from transformers import whoami

def main():
    token = os.getenv("HF_TOKEN")
    # login(token=token)
    # interpreter_login(token=token)

    compute_dtype = getattr(torch, "float16")
    # bnb_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type='nf4',
    #         bnb_4bit_compute_dtype='float16',
    #         bnb_4bit_use_double_quant=False,
    #     )
    device_map = {"": 0}

    #Download model
    model_name = "openai-community/gpt2-large"
    model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            # quantization_config=bnb_config, 
            device_map=device_map,
            trust_remote_code=True,
            use_auth_token=True
        )

    # model.config.pretraining_tp = 1 
    # peft_config = LoraConfig(
    #     r=32,
    #     lora_alpha=16,
    #     target_modules= [
    #     'q_proj',
    #     'k_proj',
    #     'v_proj',
    #     'dense',
    #     'fc1',
    #     'fc2',
    #     'embed_tokens',
    #     'lm_head'
    #     ],
    #     bias="none",
    #     lora_dropout=0.05, # Conventional
    #     task_type="CAUSAL_LM",
    # )
    # model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    training_arguments = Seq2SeqTrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=500, #CHANGE THIS IF YOU WANT IT TO SAVE LESS OFTEN. I WOULDN'T SAVE MORE OFTEN BECAUSE OF SPACE
        logging_steps=10,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        max_grad_norm=.3,
        max_steps=10000,
        warmup_ratio=.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        generation_max_length=1024,
    )

    model.config.use_cache = False

    dataset = CustomDataset("rajpurkar/squad", "train", model_name)

    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=dataset,
        data_collator=None,
        # use if lora
        # peft_config=peft_config,
        # dataset_text_field="input_ids",
        tokenizer=tokenizer,
        args=training_arguments,
        # packing=False,
    )

    trainer.train()

if __name__ == "__main__":
    main()