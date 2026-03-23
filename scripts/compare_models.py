from app.training.model_comparison import model_comparison


if __name__ == "__main__":
    result = model_comparison.compare()
    print("Model comparison generated successfully")
    print(result["summary"])