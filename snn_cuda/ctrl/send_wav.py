import wave
import serial
import sys

# Communication Parameters
SERIAL_PORTNAME = "COM9" # CHANGE ME to match your system's serial port name!
BAUD = 115200 # Make sure this matches your UART receiver

def send_wav(filename):
    """Sends the samples of a WAV file over serial port to the FPGA"""
    with wave.open(filename,"rb") as wav_file:
        assert wav_file.getnchannels() == 1, "Incorrect number of channels; re-format your WAV file!"
        assert wav_file.getsampwidth() == 1, "Incorrect sample byte-width; re-format your WAV file!"
        assert wav_file.getframerate() == 8000, "Incorrect sample rate; re-format your WAV file!"

        print(f"Opening serial port {SERIAL_PORTNAME}")
        ser = serial.Serial(SERIAL_PORTNAME,BAUD)
        
        print(f"Sending wav file {filename} bytes over serial port...")
        nframes = wav_file.getnframes()

        frames = wav_file.readframes(nframes)
        for i in range(nframes):
            sample = frames[i]
            ser.write( sample.to_bytes(1,'little') )
        print(f"wav file sent. {nframes} bytes sent")
                    


if __name__ == "__main__":
    if (len(sys.argv)<2):
        print("Usage: python3 send_wav.py <filename>")
        exit()
    filename = sys.argv[1]
    send_wav(filename)