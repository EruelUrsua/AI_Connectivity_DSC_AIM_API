from catboost import CatBoostClassifier

def load_model(file_path):
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(file_path)
    print(f"Model loaded from {file_path}")
    return loaded_model