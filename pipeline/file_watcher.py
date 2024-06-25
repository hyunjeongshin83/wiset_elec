import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

class DataChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("device_data.csv"):
            print(f"{event.src_path} has been modified.")
            # Kubeflow 파이프라인 재실행
            subprocess.run(["python", "main.py"], check=True)

if __name__ == "__main__":
    event_handler = DataChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()
    print("Monitoring for data changes...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
