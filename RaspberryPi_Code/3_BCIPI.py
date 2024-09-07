from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import time
import numpy as np
import os
import RPi.GPIO as GPIO
import tensorflow as tf

SUBJECT = "24"
MODEL = tf.keras.models.load_model(f"./DataSet/S{SUBJECT}/model_results_S{SUBJECT}/eeg_model.h5")

board_id = 38  # กำหนดค่า board id เอง เชื่อมต่อกับ MUSE 2
fs = BoardShim.get_sampling_rate(board_id)
delay = 1  # delay time in seconds
overlap = int(256 * 0.2) # 20%
N_data = fs * delay + overlap

seq_label = ["Center", "Saccades Left", "Saccades Right"]

# ตั้งค่า GPIO pin
BUZZER_PIN = 24
LED1_PIN = 26 # Left
LED2_PIN = 19 # Center
LED3_PIN = 13 # Right

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(LED1_PIN, GPIO.OUT)
GPIO.setup(LED2_PIN, GPIO.OUT)
GPIO.setup(LED3_PIN, GPIO.OUT)

def main():
    GPIO.output(LED1_PIN, GPIO.HIGH)
    GPIO.output(LED2_PIN, GPIO.HIGH)
    GPIO.output(LED3_PIN, GPIO.HIGH)
    
    board = None

    while True:
        if board is None:
            board = connect_device()
            if board is None:
                print("Cannot Connect Device. Retrying...")
                time.sleep(5)
                continue
            print("Connected to Device")

        try:
            start_using(board)
        except Exception as error:
            print(f"Error: {error}. Disconnecting and retrying...")
            board = disconnect_device(board)
            time.sleep(5)  # Wait before retrying

def start_using(board):
    index_classes = board.get_eeg_channels(board_id)
    previous_data = None

    while True:
        GPIO.output(BUZZER_PIN, GPIO.HIGH) 
        time.sleep(0.5)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        time.sleep(delay)

        data = board.get_current_board_data(N_data)
        data = data[index_classes]

        if data.shape[1] == N_data:
            for channel in range(len(index_classes)):
                DataFilter.perform_bandstop(data[channel], fs, 48.0, 52.0, 3, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                DataFilter.detrend(data[channel], DetrendOperations.LINEAR.value)
            
            data = np.expand_dims(np.transpose(data), axis=0)

            if previous_data is not None and np.array_equal(data, previous_data):
                raise Exception("Device disconnected. Data is the same as previous.")

            previous_data = data

            result = MODEL.predict(data, verbose=0)
            label_index = check_max_value(result[0])

            print_label(seq_label[label_index])
            control_led(label_index)

def connect_device():
    try:
        BoardShim.enable_dev_board_logger()

        params = BrainFlowInputParams()
        params.master_board = BoardIds.NO_BOARD
        
        board = BoardShim(board_id, params)
        board.prepare_session()
        board.start_stream()

        return board
    except:
        return None     

def disconnect_device(board):
    if board is not None:
        try:
            board.stop_stream()
            board.release_session()
        except BrainFlowError as e:
            print(f"Warning: Failed to stop streaming session. Error: {e}")
        finally:
            board.release_session()  # Ensure that the session is released
    return None


def print_label(label):
    print(f"{40*'=='}")
    print(f"\t\t{label}")
    print(f"{40*'=='}")

def control_led(index_class):
    if index_class == 0:  # Center
        GPIO.output(LED1_PIN, GPIO.LOW)
        GPIO.output(LED2_PIN, GPIO.HIGH)
        GPIO.output(LED3_PIN, GPIO.LOW)
    elif index_class == 1:  # Left
        GPIO.output(LED1_PIN, GPIO.HIGH)
        GPIO.output(LED2_PIN, GPIO.LOW)
        GPIO.output(LED3_PIN, GPIO.LOW)
    elif index_class == 2:  # Right
        GPIO.output(LED1_PIN, GPIO.LOW)
        GPIO.output(LED2_PIN, GPIO.LOW)
        GPIO.output(LED3_PIN, GPIO.HIGH)
    else:
        GPIO.output(LED1_PIN, GPIO.LOW)
        GPIO.output(LED2_PIN, GPIO.LOW)
        GPIO.output(LED3_PIN, GPIO.LOW)

def check_max_value(arr):
    max_index = np.argmax(arr)
    max_value = arr[max_index]
    if max_value >= 0.6:
        return max_index
    else:
        return 0

if __name__ == "__main__":
    print("BCI PI v 1.0 - EYE State")
    main()
    GPIO.output(LED1_PIN, GPIO.LOW)
    GPIO.output(LED2_PIN, GPIO.LOW)
    GPIO.output(LED3_PIN, GPIO.LOW)
    GPIO.cleanup()
