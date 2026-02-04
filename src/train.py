from transformers import Trainer, TrainingArguments


def train_model(model, train_dataset, val_dataset, compute_metrics):
    training_args = TrainingArguments(
           output_dir =  "./results",
            learning_rate = 2e-5,
            num_train_epochs=1,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=16,
            weight_decay = 0.01,
            logging_dir = "./logs",
            logging_steps = 100,
        eval_strategy = "no",
        save_strategy = "no",  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics = compute_metrics,
    )

    trainer.train()
    trainer.save_model("./saved_model")
    return trainer
