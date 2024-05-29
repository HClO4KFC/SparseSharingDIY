import multiprocessing
import keyboard
import time

raining=False


def worker_process(worker_id):
    print(f"Worker {worker_id} started")
    # Bind hotkey in main process
    keyboard.add_hotkey(str(worker_id), handle_hotkey, args=[worker_id])
    while True:
        time.sleep(1)


def handle_hotkey(worker_id):
    global raining
    raining = not raining
    if raining:
        print(f"{worker_id} is now raining")
    else:
        print(f"{worker_id} is no longer raining")


if __name__ == "__main__":
    num_workers = 5

    # Start worker processes
    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(target=worker_process, args=[i])
        p.start()
        processes.append(p)

    # Keep the main process running
    try:
        for proc in processes:
            proc.join()
    except KeyboardInterrupt:
        print("Exiting...")
        for p in processes:
            p.terminate()