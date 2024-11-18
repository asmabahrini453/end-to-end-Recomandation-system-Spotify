
# End-to-End Recommender System

This project focuses on building a **sequential recommendation system (RS)**. The architecture of the model depends heavily on the specific use case of the recommender system. In our case, the system predicts the next item in a sequence of items based on historical data, similar to how Spotify recommends songs for your Discover Weekly playlist based on your recent listening history.

## Problem Definition

The system takes as input a list of items (e.g., songs), and its goal is to predict the next item in the sequence. At prediction time, the input will consist of a playlist with `N` songs. The model is asked to predict the last missing item based on the first `N-1` songs.

For example, given the following playlist:

```
song_525, song_22, song_814, song_4255
```

The training set will be:

- **Query**: song_525, song_22, song_814
- **Label**: song_4255

If the model correctly predicts `song_4255`, it counts as a successful prediction.

## 1. Preparing the Dataset

After loading the data, we use **DuckDB**, an open-source, high-performance database designed for analytics workloads. DuckDB provides a fast way to process data while maintaining compatibility with Pandas DataFrames. It allows us to perform efficient data manipulation and filtering using SQL commands.

The data is then split into the following sets:

- **Training set**: Used to train the model.
- **Validation set**: Used to select the best hyperparameters.
- **Test set**: Used to evaluate the model's performance on unseen data.

## 2. Generate Embeddings and Modeling

We train an **embedding model** for the song data using the **Word2Vec** algorithm. Word2Vec is a neural network-based technique that learns associations between words (or items) by mapping them to vectors in an embedding space. These vectors capture the semantic relationships between items, making it easier to predict the next item in the sequence.

At prediction time, we use this embedding space to encode the input data and apply a **K-Nearest Neighbors (KNN)** algorithm to predict the next song.

**Hit rate** is the metric used to evaluate the performance of the model. It measures how often the model correctly predicts the next item in the sequence.

## 3. Tuning Flow

We tune multiple embedding spaces in parallel and select the best-performing one based on the validation set. After selecting the best candidate model, we test it once more on the held-out test set to evaluate its ability to generalize to unseen data.

## 4. Deployment

We use **Metaflow** in combination with **AWS** as the data store and **SageMaker** for model deployment. Metaflow provides an elegant way to manage artifacts and workflows, and SageMaker simplifies the deployment process.

The KNN-based model trained in this project is exported to a **TensorFlow** model using **Keras**. The model is then deployed to **SageMaker** using the `TensorFlowModel` abstraction, which allows us to run the model at scale with minimal configuration.

---

### Key Technologies:

- **DuckDB**: High-performance SQL database for data processing.
- **Word2Vec**: Embedding technique for mapping items to vectors.
- **KNN**: Algorithm used for prediction in the embedding space.
- **Metaflow**: Workflow management system used for model development and deployment.
- **AWS SageMaker**: Platform for deploying machine learning models.
