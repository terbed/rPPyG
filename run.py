from src.cam import *

buffer = []
cam = Camera(buffer)

if __name__ == "__main__":
    cam.start()