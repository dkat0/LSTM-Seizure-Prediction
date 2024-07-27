import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_tuner import HyperModel, RandomSearch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE

class SeizurePredictionModel(HyperModel):
    def __init__(self, random_seed=10):
        self.random_seed = random_seed
        self.early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        self._set_random_seed()

    def _set_random_seed(self):
        # Set random seed to ensure results are reproducible
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

    @staticmethod
    def process_data(eeg_dataframe):
        """
        Combine 23 chunks from each patient into one single chunk.
        """
        file_names = eeg_dataframe['Unnamed: 0'].tolist()
        patient_ids = list(set([file_name.split('.')[-1] for file_name in file_names]))
        assert len(patient_ids) == 500  # Ensure there are 500 unique patients

        patient_to_index = {patient: index for index, patient in enumerate(patient_ids)}

        eeg_data_combined = np.zeros((500, 178 * 23))  # Initialize combined EEG data array
        labels_combined = np.zeros(500)
        labels_chunks = np.zeros((500, 23))
        labels_dict = {patient: [] for patient in patient_ids}

        for _, row in eeg_dataframe.iterrows():
            file_name = row['Unnamed: 0']
            patient_id = file_name.split('.')[-1]
            patient_index = patient_to_index[patient_id]
            chunk_id = int(file_name.split('.')[0].split('X')[-1])
            
            start_index = (chunk_id - 1) * 178
            end_index = start_index + 178
            eeg_data_combined[patient_index, start_index:end_index] = row.values[1:-1]

            labels_dict[patient_id].append(row.values[-1])

        for patient_id, labels in labels_dict.items():
            patient_index = patient_to_index[patient_id]
            has_seizure = int(any(label == 1 for label in labels))
            labels_combined[patient_index] = has_seizure
            labels_chunks[patient_index, :] = [0 if label > 1 else label for label in labels]

        return eeg_data_combined, labels_combined, labels_chunks

    @staticmethod
    def plot_eeg_sequence(eeg_sequence):
        """
        Plot a single EEG sequence.
        """
        x = np.linspace(0, 23, 4094)
        plt.plot(x, eeg_sequence)
        plt.xlabel('Seconds')
        plt.ylabel('Microvolts')
        plt.show()

    @staticmethod
    def plot_accuracy(history):
        """
        Plot the training and validation accuracy over epochs.
        """
        history = history.history
        history.update({'epoch': list(range(len(history['val_accuracy'])))})
        history = pd.DataFrame.from_dict(history)
        best_epoch = history.sort_values(by='val_accuracy', ascending=False).iloc[0]['epoch']

        plt.figure()
        sns.lineplot(x='epoch', y='val_accuracy', data=history, label='Validation')
        sns.lineplot(x='epoch', y='accuracy', data=history, label='Training')
        plt.axhline(0.5, linestyle='--', color='red', label='Chance')
        plt.axvline(x=best_epoch, linestyle='--', color='green', label='Best Epoch')
        plt.legend(loc='best')
        plt.ylim([0.4, 1])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()

    def prepare_eeg(self, eeg_file):
        """
        Prepare EEG data from a CSV file.
        """
        eeg = pd.read_csv(eeg_file)
        eeg_data_combined, labels_combined, _ = self.process_data(eeg)
        return eeg_data_combined, labels_combined

    def build_model(self, hp, model_type):
        """
        Build a sequential model (RNN/LSTM/GRU) with hyperparameters and specified type.
        """
        model = Sequential()
        for i in range(hp.Int('num_layers', 1, 3)):
            if model_type == 'RNN':
                model.add(SimpleRNN(units=hp.Int(f'units_{i}', min_value=32, max_value=128, step=32),
                                    return_sequences=True if i < hp.Int('num_layers', 1, 3) - 1 else False))
            elif model_type == 'LSTM':
                model.add(LSTM(units=hp.Int(f'units_{i}', min_value=32, max_value=128, step=32),
                               return_sequences=True if i < hp.Int('num_layers', 1, 3) - 1 else False))
            elif model_type == 'GRU':
                model.add(GRU(units=hp.Int(f'units_{i}', min_value=32, max_value=128, step=32),
                              return_sequences=True if i < hp.Int('num_layers', 1, 3) - 1 else False))
            if hp.Boolean('dropout'):
                model.add(Dropout(rate=hp.Float(f'dropout_rate_{i}', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(
            optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
            loss='binary_crossentropy',
            metrics=['accuracy'])
        return model

    def train_model(self, model, x_train, y_train, x_val, y_val, checkpoint_filepath, epochs=20):
        """
        Train the given model.
        """
        monitor = ModelCheckpoint(
            checkpoint_filepath,
            monitor='val_accuracy',
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            save_freq='epoch'
        )

        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, callbacks=[monitor, self.early_stopping])
        self.plot_accuracy(history)
        return model

    def evaluate_model(self, model, x_test, y_test):
        """
        Evaluate the given model on the test data.
        """
        prob = model.predict(x_test)
        predictions = (prob > 0.5).astype(int)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    def get_best_model(self, x_train, y_train, x_test, y_test, model_type):
        """
        Perform hyperparameter tuning for the given model type and return the best model.
        """
        tuner = RandomSearch(
            hypermodel=lambda hp: self.build_model(hp, model_type),
            objective='val_accuracy',
            max_trials=10,
            executions_per_trial=1,
            directory=f'tuner_{model_type.lower()}'
        )
        tuner.search(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[self.early_stopping])
        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hps)
        return best_model

    def handle_imbalanced_data(self, x, y):
        """
        Handle imbalanced data using SMOTE.
        """
        smote = SMOTE(random_state=1)
        x_resampled, y_resampled = smote.fit_resample(x.reshape(x.shape[0], -1), y)
        x_resampled = x_resampled.reshape(-1, 23, 178)
        return x_resampled, y_resampled

    def run(self, eeg_file):
        """
        Run the complete workflow: data preparation, model training, and evaluation.
        """
        eeg_data_combined, labels_combined = self.prepare_eeg(eeg_file)

        x = eeg_data_combined.reshape(-1, 23, 178).astype(np.float32)
        y = labels_combined.astype(int).reshape(-1, 1)

        # Handle imbalanced data using SMOTE
        x_resampled, y_resampled = self.handle_imbalanced_data(x, y)

        train_indices, test_indices = train_test_split(np.arange(x_resampled.shape[0]), test_size=0.2, random_state=1)
        x_train, y_train = x_resampled[train_indices], y_resampled[train_indices]
        x_test, y_test = x_resampled[test_indices], y_resampled[test_indices]

        # Perform hyperparameter tuning
        best_rnn_model = self.get_best_model(x_train, y_train, x_test, y_test, 'RNN')
        best_lstm_model = self.get_best_model(x_train, y_train, x_test, y_test, 'LSTM')
        best_gru_model = self.get_best_model(x_train, y_train, x_test, y_test, 'GRU')
        
        # Train models
        self.train_model(best_rnn_model, x_train, y_train, x_test, y_test, './best_rnn_model.hdf5')
        self.train_model(best_lstm_model, x_train, y_train, x_test, y_test, './best_lstm_model.hdf5')
        self.train_model(best_gru_model, x_train, y_train, x_test, y_test, './best_gru_model.hdf5')

        # Print model summaries
        best_rnn_model.summary()
        best_lstm_model.summary()
        best_gru_model.summary()

        # Evaluate models on test set
        rnn_accuracy = self.evaluate_model(best_rnn_model, x_test, y_test)
        lstm_accuracy = self.evaluate_model(best_lstm_model, x_test, y_test)
        gru_accuracy = self.evaluate_model(best_gru_model, x_test, y_test)

        return rnn_accuracy, lstm_accuracy, gru_accuracy

if __name__ == '__main__':
    model = SeizurePredictionModel()

    # Run the model with the specified dataset
    rnn_acc, lstm_acc, gru_acc = model.run('./UCI Epileptic Seizure Recognition.csv')

    # Print results
    print('Results:')
    print(f'Best RNN Accuracy: {rnn_acc}')
    print(f'Best LSTM Accuracy: {lstm_acc}')
    print(f'Best GRU Accuracy: {gru_acc}')
