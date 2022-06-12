from ray.util.queue import Queue


class QueueStorage(object):
    def __init__(self, threshold=15, size=20):
        """Queue storage
        Parameters
        ----------
        threshold: int
            if the current size if larger than threshold, the data won't be collected
        size: int
            the size of the queue
        """
        self.threshold = threshold
        self.queue = Queue(maxsize=size)

    def push(self, batch):
        if self.queue.qsize() <= self.threshold:
            self.queue.put(batch)

    def pop(self):
        if self.queue.qsize() > 0:
            return self.queue.get()
        else:
            return None

    def get_len(self):
        return self.queue.qsize()
