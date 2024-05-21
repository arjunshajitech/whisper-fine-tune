from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from huggingface_hub import login
from datasets import Audio

# Login to Hugging Face | replace with your hf token
login(token="YOUR_HF_TOKEN")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('----------- device ------ : ' + device)

# Load datasets
common_voice = DatasetDict()
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "ml", split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "ml", split="test", use_auth_token=True)

# Remove unnecessary columns
common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

# Initialize feature extractor and tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Malayalam", task="transcribe")

# Initialize processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Malayalam", task="transcribe")

# Cast audio column
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# Function to prepare dataset
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# Apply preparation function
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)

# Initialize model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Update model generation configuration
model.generation_config.language = "malayalam"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.to(device)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Load WER metric
metric = evaluate.load("wer")

# Function to compute metrics
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-ml",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    # Add CUDA usage if available
    device=device
)

# Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Train the model
trainer.train()

# Push model to the Hugging Face Hub
kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_11_0",
    "dataset": "Common Voice 11.0",
    "dataset_args": "config: ml, split: test",
    "language": "ml",
    "model_name": "Whisper Small Hi - Arjun Shaji",
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
}

trainer.push_to_hub(**kwargs)

# Load the fine-tuned model and processor
model = WhisperForConditionalGeneration.from_pretrained("arjunshajitech/whisper-small-ml")
processor = WhisperProcessor.from_pretrained("arjunshajitech/whisper-small-ml")
