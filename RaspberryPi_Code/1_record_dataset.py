from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

import time

import numpy as np
import os

# art text
from art import *

# GPIO
import RPi.GPIO as GPIO

board_id = 38  # กำหนดค่า board id เอง เชื่อมต่อกับ MUSE 2

fs = BoardShim.get_sampling_rate(board_id)

delay = 1  # delay time in seconds
overlap = int(256 * 0.2) # 20%

N_DURATION = 1 
N_SAMPLE = 30

N_data = fs * delay + overlap

seq_states = ["Saccades Left", "Center", "Saccades Right"]
# seq_states = ["Center", "Saccades Left", "Saccades Right"]

# ตั้งค่า GPIO pin
BUZZER_PIN = 24
# กำหนดโหมดของการใช้ GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

def main():
    
    while True:
        choice = input(f"""
        ==> Current Parameter
        board id   : {board_id}
        fs         : {fs} Hz
        delay      : {delay} sec
        overlap    : {overlap / fs} sec
        N_DURATION : {N_DURATION} loop/class
        N sample   : {N_SAMPLE} set
        N data     : {N_data} Hz/class

        seq label  : {seq_states}

        Estimated period ==> {((delay * N_DURATION * N_SAMPLE) + (3 * len(seq_states)))  * 2} sec, {(((delay * N_DURATION * N_SAMPLE) + (3 * 2))  * 2) / 60} min
        
        ==> Menu
        1 : connect device
        2 : start recording
        3 : show example
        4 : disconnect device
        other : exit
        INPUT : """)

        if choice == "1":
            board = connect_device()
            if board == None:
                print("Cannot Connect Device\nExit....")
                break
        elif choice == "2":
            try:
                start_recording(board)
            except Exception as error:
                print(f"Exception : {error}")
                print("Cannot Record Data\nExit....")
                break
        elif choice == "3":
            show_example()
        elif choice == "4":
            try:
                disconnect_device(board)
            except:
                print("Cannot Disconnect Device\nExit....")
                break
        else:
            try:
                disconnect_device(board)
            except:
                break

def start_recording(board):
    # name_classes = board.get_eeg_names(board_id)
    index_classes = board.get_eeg_channels(board_id)
    
    filename = input("Input File Name : ")

    X_raw = []
    Y = []

    print("Setup finished, start receiving data...")
    
    for _ in range(N_SAMPLE): # loop sample
        print(f"{10*'=='} Loop : {_ + 1} {10*'=='}")
        print_label("STOP")
        input("Press any key to continue")
        
        for label in seq_states: # loop states

            print_label(label) # print label to show class recording
            
            # play sound record
            time.sleep(0.5)
            GPIO.output(BUZZER_PIN, GPIO.HIGH) 
            time.sleep(0.5)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
        
            # time.sleep(2)
            
            count_duration = 0
            while count_duration < N_DURATION:
                time.sleep(delay) # wait data internal buffer
                data = board.get_current_board_data(N_data) # get latest 256 packages or less, doesnt remove them from internal
                
                # raw_data = data[index_classes] # raw EEG data
                data = data[index_classes] # select EEG data
                
                if data.shape[1] == N_data:
    
                    for channel in range(len(index_classes)):

                        # DataFilter.perform_bandpass(data[channel], fs, 8, 100, 4, FilterTypes.BUTTERWORTH.value, 0) # filter frequency 8-100Hz
                        # Notch Filter 50 Hz
                        DataFilter.perform_bandstop(data[channel], fs, 48.0, 52.0, 3, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                        # Detrend
                        DataFilter.detrend(data[channel], DetrendOperations.LINEAR.value)
                        
                    X_raw.append(np.transpose(data))
                    
                    Y.append(label)
                    
                    count_duration += 1

            # play some sound to chang label record
            print("change class recording")
            time.sleep(1)


    # Convert lists to numpy arrays
    X_raw = np.array(X_raw)
    Y = np.array(Y)
    
    print(f"{10*'=='} Check Data {10*'=='}")
    print(f"""Shape Data
          X_raw   : {X_raw.shape}
                Y : {Y.shape}""")
    np.save(f"X_raw_{filename}.npy", X_raw)
    
    np.save(f"Y_{filename}.npy", Y)
    
    print(f"""Data Saved to 
                X_raw_{filename}.npy,
            and Y_{filename}.npy""")

def show_example():
    
    filename = input("Input File Name : ")


    print("Setup finished, start receiving data...")
    
    for _ in range(2): # loop sample
        print(f"{10*'=='} Loop : {_ + 1} {10*'=='}")
        print_label("STOP")
        input("Press any key to continue")
        
        for label in seq_states: # loop states

            print_label(label) # print label to show class recording
            
            # play sound record
            time.sleep(0.5)
            GPIO.output(BUZZER_PIN, GPIO.HIGH) 
            time.sleep(0.5)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
        
            # time.sleep(2)
            
            count_duration = 0
            while count_duration < N_DURATION:
                time.sleep(delay) # wait data internal buffer    
                count_duration += 1

            # play some sound to chang label record
            print("change class recording")
            time.sleep(1)
        
def connect_device():
    try:
        BoardShim.enable_dev_board_logger()
    
        # กำหนดค่า parameters เอง
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
    
        # name_classes = board.get_eeg_names(board_id)
        # index_classes = board.get_eeg_channels(board_id)
        
        board.prepare_session()
        
        board.start_stream()

        return board
    except:
        return None

def disconnect_device(board):
    if board != None:
        board.stop_stream()
        board.release_session()

def print_label(label):
    print(f"{40*'=='}")
    tprint(f"\t\t{label}", font="big")
    print(f"{40*'=='}")


if __name__ == "__main__":
    tprint("BCI PI v 1.0", font="big")
    tprint("EYE State", font="small")
    main()
    GPIO.cleanup()