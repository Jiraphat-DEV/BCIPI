from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

import time

import numpy as np
import os

# art text
from art import *

# GPIO
import RPi.GPIO as GPIO

# model
import tensorflow as tf

SUBJECT = "24"
# MODEL = tf.keras.models.load_model("./model_results_EYE_1/eeg_classification_model.h5")
# MODEL = tf.keras.models.load_model(f"./S{SUBJECT}/model_results_S{SUBJECT}/eeg_model.h5")
MODEL = tf.keras.models.load_model(f"./DataSet/S{SUBJECT}/model_results_S{SUBJECT}/eeg_model.h5")

# stop signal
import signal

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


# กำหนดโหมดของการใช้ GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(LED1_PIN, GPIO.OUT)
GPIO.setup(LED2_PIN, GPIO.OUT)
GPIO.setup(LED3_PIN, GPIO.OUT)

def main():
    GPIO.output(LED1_PIN, GPIO.HIGH)
    GPIO.output(LED2_PIN, GPIO.HIGH)
    GPIO.output(LED3_PIN, GPIO.HIGH)
    while True:
        choice = input(f"""
        ==> Menu
        1 : connect device
        2 : start using
        3 : start testing # not allow
        4 : disconnect device
        other : exit
        INPUT : """)

        if choice == "1": # connect device
            board = connect_device()
            if board == None:
                print("Cannot Connect Device\nExit...")
            else:
                print("Connect Device Success\nPass...")
        elif choice == "2": # start using
            try:
                start_using(board)
            except Exception as error:
                print(f"Cannot Using~!\nException : {error}\nExit...")
        elif choice == "3": # start testing
            pass
        elif choice == "4": # disconnect device
            try:
                board = disconnect_device(board)
            except Exception as error:
                print(f"Cannot Disconnect~!\nException : {error}\nExit...")
        else:
            try: # disconnect and exit
                
                board = disconnect_device(board)
                break
            except Exception as error:
                print(f"Cannot Exit~!\nException : {error}\nExit...")

def start_using(board):
    index_classes = board.get_eeg_channels(board_id)
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("You pressed Ctrl+C! Stopping the loop.")
        running = False

    def check_max_value(arr):
        max_index = np.argmax(arr)  # หาตำแหน่งของค่าสูงสุด
        max_value = arr[max_index]  # ค่าสูงสุดใน array
        if max_value >= 0.5:
            return max_index
        else:
            return 0

    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while running:
            # play sound record
            time.sleep(0.5)
            GPIO.output(BUZZER_PIN, GPIO.HIGH) 
            time.sleep(0.5)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            
            time.sleep(delay) # wait data internal buffer
            data = board.get_current_board_data(N_data) # get latest 256 packages or less, doesnt remove them from internal
            data = data[index_classes] # select EEG data

            if data.shape[1] == N_data:
                for channel in range(len(index_classes)):
                    # Notch Filter 50 Hz
                    DataFilter.perform_bandstop(data[channel], fs, 48.0, 52.0, 3, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                    # Detrend
                    DataFilter.detrend(data[channel], DetrendOperations.LINEAR.value)
                
                data = np.expand_dims(np.transpose(data), axis=0)
                # print(data.shape)

                result = MODEL.predict(data, verbose=0)
                # print(result[0])
                # print(np.argmax(result[0]))
                # print(check_max_value(result[0]))
    
                print_label(seq_label[check_max_value(result[0])])
                control_led(check_max_value(result[0]))

    except KeyboardInterrupt:
        print(f"Exit process using~!")

        
                
def connect_device():
    try:
        BoardShim.enable_dev_board_logger()
    
        params = BrainFlowInputParams()
        params.ip_port = 0
        params.serial_port = ''
        params.mac_address = ''
        params.other_info = ''
        params.serial_number = ''
        params.ip_address = ''
        params.ip_protocol = 0
        params.timeout = 15
        params.file = ''
        params.master_board = BoardIds.NO_BOARD
        
        board = BoardShim(board_id, params)
        
        board.prepare_session()
        
        board.start_stream()

        return board
    except:
        return None     

def disconnect_device(board):
    if board != None:
        board.stop_stream()
        board.release_session()

    return None

def print_label(label):
    print(f"{40*'=='}")
    tprint(f"\t\t{label}", font="big")
    print(f"{40*'=='}")

def control_led(index_class):
    if index_class == 0: # Center
        GPIO.output(LED1_PIN, GPIO.LOW)
        GPIO.output(LED2_PIN, GPIO.HIGH)
        GPIO.output(LED3_PIN, GPIO.LOW)
    elif index_class == 1: # Left
        GPIO.output(LED1_PIN, GPIO.HIGH)
        GPIO.output(LED2_PIN, GPIO.LOW)
        GPIO.output(LED3_PIN, GPIO.LOW)
    elif index_class == 2: # Left
        GPIO.output(LED1_PIN, GPIO.LOW)
        GPIO.output(LED2_PIN, GPIO.LOW)
        GPIO.output(LED3_PIN, GPIO.HIGH)
    else:
        GPIO.output(LED1_PIN, GPIO.LOW)
        GPIO.output(LED2_PIN, GPIO.LOW)
        GPIO.output(LED3_PIN, GPIO.LOW)

if __name__ == "__main__":
    tprint("BCI PI v 1.0", font="big")
    tprint("EYE State", font="small")
    main()
    # close LED
    GPIO.output(LED1_PIN, GPIO.LOW)
    GPIO.output(LED2_PIN, GPIO.LOW)
    GPIO.output(LED3_PIN, GPIO.LOW)
    GPIO.cleanup()