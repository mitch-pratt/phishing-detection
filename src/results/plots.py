import matplotlib.pyplot as plt

def plot_metric(results, metric):
    if metric not in next(iter(results.values())):
        raise ValueError(f"Metric '{metric}' not found in results")
    models = list(results.keys())
    values = [results[m][metric] for m in models]

    plt.bar(models, values)
    plt.ylabel(metric.capitalize())
    plt.title(f"Model Comparison by {metric.capitalize()}")
    plt.ylim(0, 1)
    plt.show()

def plot_metric_select(results, metric, models_to_plot=None):
    #just plot all models if none selected
    if models_to_plot is None:
        models = list(results.keys())
    else:
        models = models_to_plot

    if metric not in next(iter(results.values())):
        raise ValueError(f"Metric '{metric}' not found in results")
    values = [results[m][metric] for m in models]

    plt.bar(models, values)
    plt.ylabel(metric.capitalize())
    plt.title(f"Model Comparison by {metric.capitalize()}")
    plt.ylim(0, 1)
    plt.show()

def plot_accuracy(results):
    print("Plotting accuracy...")
    model_names = []
    accuracies = []
    for model_name in results:
        model_names.append(model_name)
        accuracies.append(results[model_name]["accuracy"])
    plt.figure()
    plt.bar(model_names, accuracies)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.show(block=True)
