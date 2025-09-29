import cv2
import numpy as np
from SDK_USER_PERMISSIONS import *
from time import sleep

# SETUP
def setup(func, *args, delay=1, success_code=0, description=''):
    '''run camera setup fucntion until it succeeds'''

    result = func(*args)
    while result != success_code:
        sleep(delay)
        result = func(*args)
    print(f'{description} Set')


if __name__ == "__main__":

    myCam = CamAPI.pyClient(manualport="/dev/ttyACM0") #Boson COM port on windows, check device manager

    # Set Radiometric Parameters
    setup(myCam.bosonSetGainMode,
          FLR_BOSON_GAINMODE_E.FLR_BOSON_HIGH_GAIN,
          description='High Gain Mode')
    setup(myCam.TLinearSetControl,
          FLR_ENABLE_E.FLR_ENABLE,
          description='TLinear Mode')
    setup(myCam.sysctrlSetUsbVideoIR16Mode,
          FLR_SYSCTRL_USBIR16_MODE_E.FLR_SYSCTRL_USBIR16_MODE_TLINEAR,
          description='IR16 Mode')
    setup(myCam.radiometrySetTransmissionWindow,
          100,
          description='Window Transimission')
    setup(myCam.TLinearRefreshLUT,
          FLR_BOSON_GAINMODE_E.FLR_BOSON_HIGH_GAIN,
          description='LUT Refresh')
    setup(myCam.bosonRunFFC,
          description='Flat Field Correction')


'''
success = 3
while success != 0:
    success = myCam.bosonSetGainMode(FLR_BOSON_GAINMODE_E.FLR_BOSON_HIGH_GAIN) #Low gain mode also available if tem>
    sleep(1)
print('Gain Mode Set')

success = 3
while success != 0:
    success = myCam.TLinearSetControl(FLR_ENABLE_E.FLR_ENABLE)
    sleep(1)
print('TLinear Mode Set')

success = 3
while success != 0:
    success = myCam.sysctrlSetUsbVideoIR16Mode(FLR_SYSCTRL_USBIR16_MODE_E.FLR_SYSCTRL_USBIR16_MODE_TLINEAR)
    sleep(1)
print(f'IR16 Mode Mode')

success = 3
while success != 0:
    success = myCam.radiometrySetTransmissionWindow(100) #100% transmission (no window in front of Boson)
    sleep(1)
print(f'Window Mode Set')

success = 3
while success != 0:
    success = myCam.TLinearRefreshLUT(FLR_BOSON_GAINMODE_E.FLR_BOSON_HIGH_GAIN) #necessary after setting radiometry>
    sleep(1)
print(f'LUT Mode Set')

success = 3
while success != 0:
    success = myCam.bosonRunFFC()
    sleep(1)
print(f'FFC Complete')



cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)     # don't auto-convert to RGB
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','1', '6', ' '))


#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
#fourcc = cv2.VideoWriter_fourcc(*'Y16 ')
#out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w, h))


while True:
    ret, frame = cap.read()
    min_val, max_val = np.min(frame), np.max(frame)
#    print(f'frame size: {frame.shape}')
#    print('Raw Values: ')
#    print(f'Min {min_val} | Max {max_val}')
#    print(f'Unique Values: {len(np.unique(frame))}')
# convert to 8bit for viewing
    if frame.dtype == np.uint16:
        print(f'\nConverting to 8bit for viewing')
        unique_vals = np.unique(frame)
        frame_8bit = ((frame-min_val)/(max_val-min_val) *255).astype(np.uint8)

# Center Temp
    center_raw = frame[int(frame.shape[0]/2), int(frame.shape[1]/2)]
    center_c = round((center_raw/100) - 273, 1)
    center_f = round(center_c * 9/5 + 32, 1)

#    print('Converted Temp Values: ')
    print(f'Center temp: Temp C - {center_c} | Temp F - {center_f}')

    cv2.imshow('Cam', frame_8bit)
    if cv2.waitKey(1)==ord('q'):
        break

cv2.destroyAllWindows()

cap.release()
'''
myCam.Close()

