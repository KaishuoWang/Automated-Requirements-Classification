# Project

## DeBERTa-v3-large
Fine-tuned DeBERTa-large model can be found at [here](https://huggingface.co/kwang123/deberta-large-ReqORNot)

| Metrics | Value |
| --- | --- |
| Accuracy | 0.9135 |
| Weighted Precision | 0.9135 |
| Weighted Recall | 0.9135 |
| Weighted F1 | 0.9134 |
| Macro Precision | 0.9135 |
| Macro Recall | 0.9128 |
| Macro F1 | 0.9131 |

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
| Accuracy | 0.8970 |
| Weighted Precision | 0.8971 |
| Weighted Recall | 0.8970 |
| Weighted F1 | 0.8971 |
| Macro Precision | 0.8969 |
| Macro Recall | 0.8971 |
| Macro F1 | 0.8970 |

## Few-shot Learning

Fine-tuned model can be found at [here](https://huggingface.co/kwang123/roberta-large-setfit-ReqORNot)

| Metrics | Value |
| --- | --- |
| Accuracy | 0.7621 |
| Weighted Precision | 0.7628 |
| Weighted Recall | 0.7621 |
| Weighted F1 | 0.7622 |
| Macro Precision | 0.7622 |
| Macro Recall | 0.7625 |
| Macro F1 | 0.7620 |

## Ensemble System

### DeBERTa + Llama2

| Metrics | Value |
| --- | --- |
| Accuracy | 0.9479 |
| Weighted Precision | 0.9711 |
| Weighted Recall | 0.9479 |
| Weighted F1 | 0.9553 |
| Macro Precision | 0.8931 |
| Macro Recall | 0.9121 |
| Macro F1 | 0.8953 |

### DeBERTa + Llama2 + Few-shot

| Metrics | Value |
| --- | --- |
| Accuracy | 0.9286 |
| Weighted Precision | 0.9637 |
| Weighted Recall | 0.9286 |
| Weighted F1 | 0.9398 |
| Macro Precision | 0.8736 |
| Macro Recall | 0.8867 |
| Macro F1 | 0.8707 |
