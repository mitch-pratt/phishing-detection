def predict_url(model, features):
    if hasattr(model, "cluster_centers_"):
        cluster = model.predict([features])[0]
        return model.cluster_label_map[cluster]
    prediction = model.predict([features])

    if hasattr(prediction, "__len__"):
        return prediction[0]
    return prediction