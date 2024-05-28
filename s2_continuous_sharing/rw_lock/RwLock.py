import multiprocessing
import time


class ReadWriteLock:
    def __init__(self):
        self._read_ready = multiprocessing.Condition(multiprocessing.Lock())
        self._readers = 0

    def acquire_read(self):
        with self._read_ready:
            self._readers += 1

    def release_read(self):
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self):
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        self._read_ready.release()


def reader(shared_data, rw_lock, reader_id):
    while True:
        rw_lock.acquire_read()
        try:
            print(f"Reader {reader_id} reading data: {shared_data['value']}")
        finally:
            rw_lock.release_read()
        time.sleep(1)


def writer(shared_data, rw_lock):
    for i in range(5):
        rw_lock.acquire_write()
        try:
            shared_data['value'] = f"New data {i}"
            print(f"Writer updated data to: {shared_data['value']}")
        finally:
            rw_lock.release_write()
        time.sleep(3)


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared_data = manager.dict()
    shared_data['value'] = "Initial data"

    rw_lock = ReadWriteLock()

    readers = [multiprocessing.Process(target=reader, args=(shared_data, rw_lock, i)) for i in range(3)]
    writer_process = multiprocessing.Process(target=writer, args=(shared_data, rw_lock))

    for r in readers:
        r.start()
    writer_process.start()

    for r in readers:
        r.join()
    writer_process.join()
