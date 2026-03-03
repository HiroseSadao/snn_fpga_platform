import serial
import wave


# opens serial port, waits for 6 seconds of 8kHz audio data, writes it to output.wav

# set to proper serial port name and WAV!
# find the port name using test_ports.py
# CHANGE ME
SERIAL_PORT_NAME = "COM9"
BAUD_RATE = 115200

ser = serial.Serial(SERIAL_PORT_NAME,BAUD_RATE)
print("Serial port initialized")

print("Recording 6 seconds of audio:")
ypoints = []
for i in range(8000*6):
    val = int.from_bytes(ser.read(),'little')
    if ((i+1)%8000==0):
        print(f"{(i+1)/8000} seconds complete")
    ypoints.append(val)

with wave.open('output.wav','wb') as wf:
    wf.setframerate(8000)
    wf.setnchannels(1)
    wf.setsampwidth(1)
    samples = bytearray(ypoints)
    wf.writeframes(samples)
    print("Recording saved to output.wav")