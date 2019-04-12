from src.cam import *
from src.load import FrameLoader


buffer = []
# cam = Camera(buffer)
loader = FrameLoader("/media/terbe/SZTAKIHDD/MMSE-HR/first_10_subjects_2D/F013/T1/", "jpg", buffer)

if __name__ == "__main__":
    # cam.start()
    loader.start()
