import os
import re
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers


EEG_DATA_PATH = './EEG'
CNN_MODELS_PATH = './trained_models'
EEG_SIGNAL_FRAME_SIZE = 10
EEG_SIGNAL_FRAME_TIME = 10
CNN_POSITIVE_LABEL = 1
CNN_NEGATIVE_LABEL = 0
CNN_TEST_SIZE = 0.2
CNN_INPUT_SHAPE = (19, EEG_SIGNAL_FRAME_SIZE * EEG_SIGNAL_FRAME_TIME, 1) # 19 - num of electrodes ; 1000 - EEG_SIGNAL_FRAME_SIZE * EEG_SIGNAL_FRAME_TIME ; 1 - constant
CNN_EPOCHS = 5 # Liczba powtórzeń z jaką sztu

def printStatus(statusText):
    print("STATUS: " + statusText)
    print("///////////////////////")


def readEEGRaw(folder_path):
    """
        Matlab files from directory to numpy 2d arrays

        :param folder_path: EEG folder file with unziped subfolders
        :return: 2 lists (ADHD_DATA, CONTROL_DATA) of 2d numpy arrays, each 19 x individual_sample_count
    """

    # Lista podfolderów
    subfolders = ['ADHD_part1', 'ADHD_part2', 'Control_part1', 'Control_part2']

    # Listy na dane
    ADHD_DATA = []
    CONTROL_DATA = []

    # Pętla po podfolderach
    for subfolder in subfolders:
        current_folder = os.path.join(folder_path, subfolder)

        # Lista plików .mat w bieżącym podfolderze
        mat_files = [f for f in os.listdir(current_folder) if f.endswith('.mat')]

        # Pętla po plikach .mat
        for mat_file in mat_files:
            file_path = os.path.join(current_folder, mat_file)

            # Wczytanie danych z pliku .mat
            loaded_data = loadmat(file_path)

            # Uzyskanie nazwy pliku bez rozszerzenia
            file_name, _ = os.path.splitext(mat_file)

            # Zapisanie danych do odpowiedniego słownika w zależności od grupy
            if 'ADHD' in subfolder:
                arr = loaded_data[file_name]
                ADHD_DATA.append(arr.T)
            elif 'Control' in subfolder:
                arr = loaded_data[file_name]
                CONTROL_DATA.append(arr.T)

    return ADHD_DATA, CONTROL_DATA


def standardizeEEGData(dataList, frameSize, time):
    """
        Turn patient EEG list of data (list of 2D matrixes) to 3D matrix with set sample size

        :param  dataList: List of 2D matrixes preferable set as [number_of_EEG_electrodes][electrode_samples]
        :param  frameSize: Size of one frame of samples
        :param  time: Amount of  frames in signal 
        :return: 3D matrix of size [num_of_total_data_frames][number_of_EEG_electrodes][frameSize * time]
    """
    result = []

    for matrix in dataList:
        num_rows, num_samples = matrix.shape
        num_frames = (num_samples // (frameSize * time))

        divided_matrix = np.array_split(matrix[:, :num_frames * frameSize * time], num_frames, axis=1)
        result.extend(divided_matrix)

    return np.array(result)


def check_saved_trained_model():
    trained_model_files = [f for f in os.listdir(CNN_MODELS_PATH) if f.endswith('.h5')]
    if trained_model_files:
        # Wydobywanie precyzji z nazw plików i wybieranie największej
        max_accuracy = max([float(re.search(r"(\d+\.\d+).h5", file).group(1)) for file in trained_model_files])
        trained_model_path = os.path.join(CNN_MODELS_PATH, f'{max_accuracy:.4f}.h5')
        trained_model = keras.models.load_model(trained_model_path)
        print(f"Trained model loaded from file: {trained_model_path}")
        return trained_model
    else:
        return None
    

def save_trained_model(trained_model, final_accuracy):
    if not os.path.exists(CNN_MODELS_PATH):
        os.makedirs(CNN_MODELS_PATH)
    trained_model_path = os.path.join(CNN_MODELS_PATH, f'{final_accuracy:.4f}.h5')
    trained_model.save(trained_model_path)
    print(f"Trained model saved to file: {trained_model_path}")


# Sprawdzenie czy plik z nauczonym modelem już istnieje
trained_model = check_saved_trained_model()

#user_choice = input("Do you want to train a new model (enter 'train') or load an existing one (enter 'load')? ").lower()

user_choice = 'train'

# Jeśli model nie istnieje, wczytaj dane i stwórz nowy model
if user_choice == 'train': 
    ADHD_DATA, CONTROL_DATA = readEEGRaw(EEG_DATA_PATH)

    printStatus("FILE DATA READ")

    ADHD_MAT = standardizeEEGData(ADHD_DATA, EEG_SIGNAL_FRAME_SIZE, EEG_SIGNAL_FRAME_TIME)
    CONTROL_MAT = standardizeEEGData(CONTROL_DATA, EEG_SIGNAL_FRAME_SIZE, EEG_SIGNAL_FRAME_TIME)

    printStatus("DATA REFORMATED")

    labelList = [CNN_POSITIVE_LABEL] * len(ADHD_MAT) + [CNN_NEGATIVE_LABEL] * len(CONTROL_MAT)

    X_DATA = np.concatenate((ADHD_MAT, CONTROL_MAT), axis=0)
    Y_DATA = np.array(labelList)

    X_train, X_test, y_train, y_test = train_test_split(X_DATA, Y_DATA, test_size=CNN_TEST_SIZE, random_state=42)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    printStatus("DATA READY")

    model = keras.Sequential([
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=CNN_INPUT_SHAPE, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    printStatus("MODEL COMPILED")

    model.fit(X_train, y_train, epochs=CNN_EPOCHS, batch_size=2)
    _, final_accuracy = model.evaluate(X_test, y_test)
    print(f"Final accuracy: {final_accuracy}")

    printStatus("MODEL FITTED")

    save_trained_model(model, final_accuracy)
elif user_choice == 'load':
    # Sprawdź, czy istnieje wcześniej nauczony model
    trained_model = check_saved_trained_model()

    if trained_model is not None:
        # Jeśli model istnieje, użyj istniejącego modelu
        model = trained_model
    else:
        print("No existing trained model found. Please train a new model.")
        exit()

else:
    print("Invalid choice. Please enter 'train' or 'load'.")
    exit()