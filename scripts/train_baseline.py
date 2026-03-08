from app.training.baseline_trainer import baseline_trainer


if __name__ == "__main__":
    result = baseline_trainer.train()
    print(result["message"])
    print("Model saved to:", result["model_path"])
    print("Metrics saved to:", result["metrics_path"])
    print("Metrics:", result["metrics"])