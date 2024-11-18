####### RECOMANDATION SYSTEM FOR SPOTIFY PLAYLIST #######

#import necessary libraries
from metaflow import FlowSpec, step, S3, Parameter, current, card
from metaflow.cards import Markdown, Table
import os
import json
import time
from random import choice

#main flow class from metaflow for the RS pipeline
class RecSysDeployment(FlowSpec):
    
    #IS_DEV:controlles if the flow runs in dev mode
    IS_DEV = Parameter(
        name="is_dev",
        help="Flag for dev development, with a smaller dataset",
        default="1",
    )
    #KNN 
    KNN_K = Parameter(
        name="knn_k",
        help="Number of neighbors we retrieve from the vector space",
        default="100",
    )

    """
        Initial step to check configurations and ensure everything works correctly,
        or fail fast!
    """
    @step #@step:Tracks the function as a step in a pipeline=>ensures modularity & debugging
    def start(self):
       
        from metaflow.metaflow_config import DATASTORE_SYSROOT_S3

        # Print debug information about the current Metaflow run
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        # if you're using Metaflow with AWS, you should see an s3 url here!
        print("datastore is: %s" % DATASTORE_SYSROOT_S3)
        # Alert if running in development mode.
        if self.IS_DEV == "1":
            print("ATTENTION: RUNNING AS DEV VERSION - DATA WILL BE SUB-SAMPLED!!!")
        # Proceed to the next step with ".next"
        self.next(self.prepare_dataset)

    """
        Prepares the dataset for training by:
        - Loading a dataset file:'cleaned_spotify_dataset.parquet'
        - Transforming and shaping the data using DuckDB.
        - Splitting the dataset into training, validation, and test sets.
    """
    @step
    def prepare_dataset(self):
       
        import duckdb
        import numpy as np

        # we start a fast in-memory database
        con = duckdb.connect(database=":memory:")
        # Load the dataset
        # we create a new id for the playlist, by concatenating user and playlist name
        # since songs can have the same name (e.g. Intro), we make them (more?) unique by
        # concatenating the artist and the track with a special symbol |||
        # Load the data into the DuckDB database
        con.execute(
            """
            SET threads TO 1;
            SET memory_limit='800MB';
            SET enable_progress_bar=true;
            SET progress_bar_time=6000;
            CREATE TABLE playlists AS
            SELECT *,
            CONCAT (user_id, '-', playlist) as playlist_id,
            CONCAT (artist, '|||', track) as track_id,
            FROM 'cleaned_spotify_dataset.parquet'
            ;
        """
        )
        # inspect the first line
        con.execute("SELECT * FROM playlists LIMIT 1;")
        print(con.fetchone())
        # Using DuckDB's fast SQL interface for quick descriptive stats as an alternative
        #  to Pandas long code

        tables = ["row_id", "user_id", "track_id", "playlist_id", "artist"]
        for t in tables:
            con.execute("SELECT COUNT(DISTINCT({})) FROM playlists;".format(t))
            print("# of {}".format(t), con.fetchone()[0])

        #we are getting data in the shape we need for Prod2Vec type
        # of representation - each row is keyed by playlist id, and with two arrays
        # for the sequence of artists in the playlist and the sequence of songs:exp:
        # 9cc0cfd4d7d7885102480dd99e7a90d6-HardRock | [ artist_1, ... artist_n ] | [ song_1, ... song_n ]
        # We use the original row_id as index for the playlist ordering

        # in dev mode, we reduce the dataset size for faster iteration
        sampling_cmd = ""
        if self.IS_DEV == "1":
            print("Subsampling data, since this is DEV")
            sampling_cmd = "USING SAMPLE 10 PERCENT (bernoulli)" if self.IS_DEV == "1" else ""
        
        # build the dataset query:
        # track_test_x is the list of songs in a playlist except the LAST one
        # track_test_y is the LAST song - we will use these columns for
        # validation and testing of our recsys
        dataset_query = """
            SELECT * FROM
            (
                SELECT
                    playlist_id,
                    LIST(artist ORDER BY row_id ASC) as artist_sequence,
                    LIST(track_id ORDER BY row_id ASC) as track_sequence,
                    array_pop_back(LIST(track_id ORDER BY row_id ASC)) as track_test_x,
                    LIST(track_id ORDER BY row_id ASC)[-1] as track_test_y
                FROM
                    playlists
                GROUP BY playlist_id
                HAVING len(track_sequence) > 2
            )
            {}
            ;
            """.format(
            sampling_cmd
        )
        # Execute the query and fetch the results into a Pandas DataFrame =df
        con.execute(dataset_query)
        df = con.fetch_df()
        print("# rows: {}".format(len(df)))
        # debug: print the first row
        print(df.iloc[0].tolist())
        # close out the db connection
        con.close()
       # Split the dataset into training, validation, and testing sets
        train, validate, test = np.split(
            df.sample(frac=1, random_state=42), [int(0.7 * len(df)), int(0.9 * len(df))]
        )
       
        self.df_dataset = df
        self.df_train = train
        self.df_validate = validate
        self.df_test = test
        # Print statistics about the testing set.
        print("# testing rows: {}".format(len(self.df_test)))
      
        # Prepare hyperparameter sets for training.
        self.hypers_sets = [
            json.dumps(_)
            for _ in [
                {
                    "min_count": 3,
                    "epochs": 30,
                    "vector_size": 48,
                    "window": 10,
                    "ns_exponent": 0.75,
                },
                {
                    "min_count": 5,
                    "epochs": 30,
                    "vector_size": 48,
                    "window": 10,
                    "ns_exponent": 0.75,
                },
                {
                    "min_count": 10,
                    "epochs": 30,
                    "vector_size": 48,
                    "window": 10,
                    "ns_exponent": 0.75,
                },
            ]
        ]
       
        self.next(self.generate_embeddings, foreach="hypers_sets")


    """
        Given an embedding space from Word2Vec, predict best next song with KNN.
    """
    def predict_next_track(self, vector_space, input_sequence, k):
        
        query_item = input_sequence[-1]
        if query_item not in vector_space:
            # pick a random item instead
            query_item = choice(list(vector_space.index_to_key))

        return [_[0] for _ in vector_space.most_similar(query_item, topn=k)]


    """
        Evaluate the model using Hit Rate
    """
    def evaluate_model(self, _df, vector_space, k):
        lambda_predict = lambda row: self.predict_next_track(
            vector_space, row["track_test_x"], k
        )
        _df["predictions"] = _df.apply(lambda_predict, axis=1)
        lambda_hit = lambda row: 1 if row["track_test_y"] in row["predictions"] else 0
        _df["hit"] = _df.apply(lambda_hit, axis=1)
        hit_rate = _df["hit"].sum() / len(_df)
        return hit_rate
    
    """
        generate vector representations for songs using Word2Vec
    """
    @step
    def generate_embeddings(self):
    
        from gensim.models.word2vec import Word2Vec

      
        self.hyper_string = self.input
        self.hypers = json.loads(self.hyper_string)
        track2vec_model = Word2Vec(self.df_train["track_sequence"], **self.hypers)
        print("Training with hypers {} is completed!".format(self.hyper_string))
        print("Vector space size: {}".format(len(track2vec_model.wv.index_to_key)))
        # debug with a random example
        test_track = choice(list(track2vec_model.wv.index_to_key))
        print("Example track: '{}'".format(test_track))
        test_vector = track2vec_model.wv[test_track]
        print("Test vector for '{}': {}".format(test_track, test_vector[:5]))
        test_sims = track2vec_model.wv.most_similar(test_track, topn=3)
        print("Similar songs to '{}': {}".format(test_track, test_sims))
        # calculate the validation score as hit rate
        self.validation_metric = self.evaluate_model(
            self.df_validate, track2vec_model.wv, k=int(self.KNN_K)
        )
        print("Hit Rate@{} is: {}".format(self.KNN_K, self.validation_metric))
        # finally, version the embeddings
        self.track_vectors = track2vec_model.wv

        self.next(self.join_runs)

    """
        Join the parallel runs and merge results into a dictionary.
    """
    @card(type="blank", id="hyperCard")
    @step
    def join_runs(self, inputs):
        """
        Merges results from multiple runs with different parameters
        and collects predictions made by the different versions.
        """
    # Merge results from each run, storing vectors and validation metrics in dictionaries
        self.all_vectors = {inp.hyper_string: inp.track_vectors for inp in inputs}
        self.all_results = {inp.hyper_string: inp.validation_metric for inp in inputs}
        print("Current result map: {}".format(self.all_results))
        # pick one according to best hit rate
        self.best_model, self_best_result = sorted(
            self.all_results.items(), key=lambda x: x[1], reverse=True
        )[0]
        print(
            "The best validation score is for model: {}, {}".format(
                self.best_model, self_best_result
            )
        )
        # Select the vectors of the best model as the final ones
        self.final_vectors = self.all_vectors[self.best_model]
        self.final_dataset = inputs[0].df_test # Use the test data of the first input

        current.card.append(Markdown("## Results from parallel training"))
        current.card.append(
            Table([[inp.hyper_string, inp.validation_metric] for inp in inputs])
        )
        # next, test the best model on unseen data, and report the final Hit Rate as
        # our best point-wise estimate of "in the wild" performance
        self.next(self.model_testing)
    """
    Tests the best model's generalization ability by running predictions on unseen test data.
    """
    @step
    def model_testing(self):
        # Evaluate the model using a hit rate metric
        self.test_metric = self.evaluate_model(
            self.final_dataset, self.final_vectors, k=int(self.KNN_K)
        )
        print("Hit Rate@{} on the test set is: {}".format(self.KNN_K, self.test_metric))
        self.next(self.deploy)

    def keras_model(
        self,
        all_ids: list,
        song_vectors,  # np array with vectors for songs
        test_id: str,# ID of the test song
        test_vector, # Vector of the test song
    ):
        """
        Build a retrieval model using TF recommender abstraction for song recommendations based on vectors
        """
        import tensorflow as tf
        import tensorflow_recommenders as tfrs
        import numpy as np
        # Get the embedding dimension (size of vectors)
        embedding_dimension = song_vectors[0].shape[0]
        print("Vector space dims: {}".format(embedding_dimension))
        
        # add to the existing matrix of weight a 0.0.0.0... vector for unknown items
        unknown_vector = np.zeros((1, embedding_dimension))
        print(song_vectors.shape, unknown_vector.shape)
        # Combine the vectors
        embedding_matrix = np.r_[unknown_vector, song_vectors]

        # first item is the unknown token! it is all 0
        print(embedding_matrix.shape)
        assert embedding_matrix[0][0] == 0.0
        # Create an embedding layer and set its weights to the embedding matrix
        embedding_layer = tf.keras.layers.Embedding(
            len(all_ids) + 1, embedding_dimension
        )
        embedding_layer.build((None,))# Build the layer
        embedding_layer.set_weights([embedding_matrix]) # Set the weights
        embedding_layer.trainable = False # Freeze the weights (no training)


        # Create a model to look up song vectors based on song IDs
        vector_model = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=all_ids, mask_token=None), # Map song IDs to indices
                embedding_layer,
            ]
        )
        _v = vector_model(np.array([test_id]))# Get vector for the test song
       # Print the first few elements of the test vector
        print(test_vector[:3])
        # Print the first few elements of the model's prediction
        print(_v[0][:3])
        # test unknonw ID
        print("Test unknown id:")
        print(vector_model(np.array(["blahdagkagda"]))[0][:3])
        
         # Create a retrieval model using the index of songs
        song_index = tfrs.layers.factorized_top_k.BruteForce(vector_model)
        song_index.index(song_vectors, np.array(all_ids))# Index the song vectors
        # Try a prediction for the test song
        _, names = song_index(tf.constant([test_id]))
        print(f"Recommendations after track '{test_id}': {names[0, :3]}")

        return song_index

    """
    Builds a retrieval model and saves it for serving.
    """
    def build_retrieval_model(self):
        
        # generate a signature for the endpointand timestamp as a convention
        self.model_timestamp = int(round(time.time() * 1000))
        # save model: TF models need to have a version
        model_name = "playlist-recs-model/1"# Model version

        # pick one song , as index, to use as a test
        self.test_index = 3
        retrieval_model = self.keras_model(
            self.all_ids,
            self.startup_embeddings,
            self.all_ids[self.test_index],
            self.startup_embeddings[self.test_index],
        )
        # Save the model for serving
        retrieval_model.save(filepath=model_name)
    """
    Deploys the trained model and sets up for serving predictions.
    """
    @step
    def deploy(self):
      
        import numpy as np

         # Get the list of song IDs and their corresponding vectors
        self.all_ids = list(self.final_vectors.index_to_key)
        self.startup_embeddings = np.array(
            [self.final_vectors[_] for _ in self.all_ids]
        )
        # Build the retrieval model and save it
        self.build_retrieval_model()
        self.next(self.end)

    @step
    def end(self):
        print(
            """
        Now you are ready to start your TensorFlow Serving endpoint! 🎉 🥳
        Run the following command in a VSCode terminal:
        \nnohup tensorflow_model_server --rest_api_port=8501 --model_name="${TF_MODEL_NAME}" --model_base_path="${TF_MODEL_DIR}" >server.log 2>&1
        \nThe endpoint will be available at http://localhost:8501/v1/models/${TF_MODEL_NAME}:predict
        \nWithout closing the first terminal, open a new one.
        Now you can request a prediction from the endpoint 🤖 🧠
        \nAfter starting the TensorFlow server, try to recommend new tracks from your terminal:
        \npython predict.py --track "Coldplay|||The Scientist"
        python predict.py --track "The Rolling Stones|||Wild Horses"
        python predict.py --track "Rihanna|||We Found Love"
        """
        )
        return


if __name__ == "__main__":
    RecSysDeployment()
