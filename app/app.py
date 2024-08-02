from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load models
knn = joblib.load('models/knn_model.pkl')
nb = joblib.load('models/nb_model.pkl')
kmeans = joblib.load('models/kmeans_model.pkl')
gmm = joblib.load('models/gmm_model.pkl')

# Initialize scaler
scaler = StandardScaler()

# Load accuracy metrics
metrics_df = pd.read_csv('data/model_metrics.csv')
model_accuracies = metrics_df.set_index('Model').to_dict()['Accuracy']


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        transaction_id = int(request.form['transaction_id'])
        name = request.form['name']
        place = request.form['place']
        transaction_amount = float(request.form['transaction_amount'])
        transaction_time = float(request.form['transaction_time'])

        # Prepare input data
        X_input = pd.DataFrame([[transaction_amount, transaction_time]], columns=[
                               'transaction_amount', 'transaction_time'])
        X_input = scaler.fit_transform(X_input)  # Ensure consistent scaling

        # Predictions from all models
        knn_pred = knn.predict(X_input)[0]
        nb_pred = nb.predict(X_input)[0]
        # This may require adjustments based on K-Means output
        kmeans_pred = kmeans.predict(X_input)[0]
        # This may require adjustments based on GMM output
        gmm_pred = gmm.predict(X_input)[0]

        # Save predictions to CSV
        results_df = pd.DataFrame({
            'transaction_id': [transaction_id],
            'name': [name],
            'place': [place],
            'transaction_amount': [transaction_amount],
            'transaction_time': [transaction_time],
            'knn_prediction': [knn_pred],
            'nb_prediction': [nb_pred],
            'kmeans_prediction': [kmeans_pred],
            'gmm_prediction': [gmm_pred]
        })
        results_df.to_csv('data/predictions.csv', mode='a',
                          header=False, index=False)

        # Get accuracy metrics for display
        knn_accuracy = model_accuracies.get('KNN', 'N/A')
        nb_accuracy = model_accuracies.get('Naive Bayes', 'N/A')

        return render_template('index.html',
                               knn_pred=knn_pred,
                               nb_pred=nb_pred,
                               kmeans_pred=kmeans_pred,
                               gmm_pred=gmm_pred,
                               transaction_id=transaction_id,
                               name=name,
                               place=place,
                               transaction_amount=transaction_amount,
                               transaction_time=transaction_time,
                               knn_accuracy=knn_accuracy,
                               nb_accuracy=nb_accuracy,
                               accuracy_bar_graph_url='/static/accuracy_bar_graph.png',
                               scatter_plot_url='/static/3d_scatter_plot.png')

    return render_template('index.html',
                           knn_pred=None,
                           nb_pred=None,
                           kmeans_pred=None,
                           gmm_pred=None,
                           knn_accuracy='N/A',
                           nb_accuracy='N/A',
                           accuracy_bar_graph_url='/static/accuracy_bar_graph.png',
                           scatter_plot_url='/static/3d_scatter_plot.png')


if __name__ == '__main__':
    app.run(debug=True)
