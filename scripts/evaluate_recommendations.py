from app.training.recommendation_evaluator import recommendation_evaluator


if __name__ == "__main__":
    result = recommendation_evaluator.evaluate()
    print("Recommendation evaluation completed successfully")
    print(result)