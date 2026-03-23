
from app.training.dynamic_gnn_trainer import dynamic_gnn_training_pipeline


if __name__ == "__main__":
    result = dynamic_gnn_training_pipeline.train()
    print(result["message"])
    print("Model saved to:", result["model_path"])
    print("Metrics saved to:", result["metrics_path"])
    print("Final validation accuracy:", result["final_val_accuracy"])