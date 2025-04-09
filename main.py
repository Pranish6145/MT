from utils.motion_detector import MotionDetector

if __name__ == "__main__":
    detector = MotionDetector(source=0)  # 0 = webcam, or 'videos/sample.mp4'
    detector.run()
