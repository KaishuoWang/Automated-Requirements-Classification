import numpy as np
import pandas as pd
import evaluate
from setfit import sample_dataset, SetFitModel
from setfit import Trainer as SetFitTrainer
from setfit import TrainingArguments as SetFitTrainingArguments
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import precision_score, recall_score, f1_score
import bitsandbytes
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class DeBERTaModel:
    def __init__(self, model_name, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def preprocess(self, examples):
        return self.tokenizer(examples['text'], truncation=True)

    def train(self, data, fold, push_to_hub, epochs=3, batch_size=4):
        tokenized_data = data.map(self.preprocess, batched=True, remove_columns=['text'])

        training_args = TrainingArguments(
            output_dir=f"deberta-ReqORNot-{fold}",
            overwrite_output_dir=True,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model='eval_macro f1'
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        if push_to_hub:
            trainer.push_to_hub(f"deberta-ReqORNot-{fold}")
        # trainer.save_model('llama2-' + fold)
        # self.tokenizer.save_pretrained('llama2-' + fold)

        return trainer.evaluate()

    def compute_metrics(self, eval_pred):
        accuracy = evaluate.load("accuracy")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        matrics = accuracy.compute(predictions=predictions, references=labels)

        matrics['weighted precision'] = precision_score(labels, predictions, average='weighted')
        matrics['weighted recall'] = recall_score(labels, predictions, average='weighted')
        matrics['weighted f1'] = f1_score(labels, predictions, average='weighted')

        matrics['macro precision'] = precision_score(labels, predictions, average='macro')
        matrics['macro recall'] = recall_score(labels, predictions, average='macro')
        matrics['macro f1'] = f1_score(labels, predictions, average='macro')
        return matrics
    

class LlamaModel:
    def __init__(self, model_name, hf_token):
        self.hf_token = hf_token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self.load_model(model_name)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.hf_token = hf_token

    # def print_trainable_parameters(self, model):
    #     """
    #     Prints the number of trainable parameters in the model.
    #     """
    #     trainable_params = 0
    #     all_param = 0
    #     for _, param in model.named_parameters():
    #         all_param += param.numel()
    #         if param.requires_grad:
    #             trainable_params += param.numel()
    #     print(
    #         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    #     )
    
    def find_all_linear_names(self, model):
        """
        Find modules to apply LoRA to.

        :param model: PEFT model
        """

        cls = bitsandbytes.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        print(f"LoRA module names: {list(lora_module_names)}")
        return list(lora_module_names)

    def load_model(self, model_name):
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            token=self.hf_token,
            num_labels=2,
        )

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=self.find_all_linear_names(model),
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS"
        )

        model = get_peft_model(model, config)
        # self.print_trainable_parameters(model)

        return model

    def preprocess(self, examples):
        return self.tokenizer(examples['text'], max_length=4096, truncation=True)

    def train(self, data, fold, push_to_hub, epochs=5, batch_size=30):
        tokenized_data = data.map(self.preprocess, batched=True, remove_columns=['text'])

        training_args = TrainingArguments(
            output_dir=f"llama2-ReqORNot-{fold}",
            overwrite_output_dir=True,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model='eval_macro f1'
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.model.config.use_cache = False

        trainer.train()

        if push_to_hub:
            trainer.push_to_hub(f"llama2-ReqORNot-{fold}")
        # trainer.save_model('llama2-' + fold)
        # self.tokenizer.save_pretrained('llama2-' + fold)

        return trainer.evaluate()

    def compute_metrics(self, eval_pred):
        accuracy = evaluate.load("accuracy")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        matrics = accuracy.compute(predictions=predictions, references=labels)

        matrics['weighted precision'] = precision_score(labels, predictions, average='weighted')
        matrics['weighted recall'] = recall_score(labels, predictions, average='weighted')
        matrics['weighted f1'] = f1_score(labels, predictions, average='weighted')

        matrics['macro precision'] = precision_score(labels, predictions, average='macro')
        matrics['macro recall'] = recall_score(labels, predictions, average='macro')
        matrics['macro f1'] = f1_score(labels, predictions, average='macro')
        return matrics
    

class FewShotModel:
    def __init__(self, model_name, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = SetFitModel.from_pretrained(model_name).to(device)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train(self, data, fold, push_to_hub, batch_size=8):
        train_dataset = sample_dataset(data["train"], num_samples=24)
        eval_dataset = data["test"]

        args = SetFitTrainingArguments(
            batch_size=batch_size,
            num_epochs=10,
        )

        trainer = SetFitTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            metric=self.compute_metrics,
        )
        trainer.train()

        if push_to_hub:
            trainer.push_to_hub(f"few-shot-ReqORNot-{fold}")
        # trainer.save_model('few-shot-' + fold)
        # self.tokenizer.save_pretrained('few-shot-' + fold)

        return trainer.evaluate()
    
    def compute_metrics(self, predictions, labels):
        accuracy = evaluate.load("accuracy")
        matrics = accuracy.compute(predictions=predictions, references=labels)

        matrics['weighted precision'] = precision_score(labels, predictions, average='weighted')
        matrics['weighted recall'] = recall_score(labels, predictions, average='weighted')
        matrics['weighted f1'] = f1_score(labels, predictions, average='weighted')

        matrics['macro precision'] = precision_score(labels, predictions, average='macro')
        matrics['macro recall'] = recall_score(labels, predictions, average='macro')
        matrics['macro f1'] = f1_score(labels, predictions, average='macro')
        return matrics