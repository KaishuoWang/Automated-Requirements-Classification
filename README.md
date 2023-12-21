# Project

## DeBERTa-v3-large
Fine-tuned DeBERTa-large model can be found at [here](https://huggingface.co/kwang123/deberta-large-ReqORNot)

| Metrics | Value |
| --- | --- |
| Accuracy | 0.939142 |
| Weighted Precision | 0.939169	 |
| Weighted Recall | 0.939142 |
| Weighted F1 | 0.939134 |
| Macro Precision | 0.939237 |
| Macro Recall | 0.938984 |
| Macro F1 | 0.939089 |

## Llama2-7b
Fine-tuned Llama2-7b model can be found at [here](https://huggingface.co/kwang123/llama2-7b-ReqORNot)

Note this is just a adapter, you still need to have access to the original Llama2-7b model in order to use it.

The model can be load by
```python
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
    token='YOUR HUGGINGFACE ACCESS TOKEN HERE',
    num_labels=2,
)
```

| Metrics | Value |
| --- | --- |
| Accuracy | 0.944532 |
| Weighted Precision | 0.945637	 |
| Weighted Recall | 0.944532 |
| Weighted F1 | 0.944740 |
| Macro Precision | 0.939452 |
| Macro Recall | 0.946108 |
| Macro F1 | 0.942408 |

## Few-shot Learning

Fine-tuned model can be found at [here](https://huggingface.co/kwang123/roberta-large-setfit-ReqORNot)

| Metrics | Value |
| --- | --- |
| Accuracy | 0.741355 |
| Weighted Precision | 0.769921	 |
| Weighted Recall | 0.741355 |
| Weighted F1 | 0.737959 |
| Macro Precision | 0.763828 |
| Macro Recall | 0.750018 |
| Macro F1 | 0.739408 |

## Ensemble System

### DeBERTa + Llama2

| Metrics | Value |
| --- | --- |
| Accuracy | 0.955729 |
| Weighted Precision | 0.964219	 |
| Weighted Recall | 0.955729 |
| Weighted F1 | 0.957056 |
| Macro Precision | 0.938220 |
| Macro Recall | 0.956013 |
| Macro F1 | 0.942336 |

### DeBERTa + Llama2 + Few-shot

| Metrics | Value |
| --- | --- |
| Accuracy | 0.945312 |
| Weighted Precision | 0.956053	 |
| Weighted Recall | 0.945312 |
| Weighted F1 | 0.946927 |
| Macro Precision | 0.919577 |
| Macro Recall | 0.930135 |
| Macro F1 | 0.919556 |
