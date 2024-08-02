import pandas as pd
import matplotlib.pyplot as plt


def generate_accuracy_bar_graph():
    # Load accuracy metrics
    metrics_df = pd.read_csv('./data/model_metrics.csv')

    # Create a bar graph for model accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_df['Model'], metrics_df['Accuracy'],
            color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save plot as an image file
    plt.savefig('./app/static/accuracy_bar_graph.png')
    plt.close()
    print("Accuracy bar graph successfully saved as 'accuracy_bar_graph.png'")


generate_accuracy_bar_graph()
