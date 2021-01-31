import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        total_time = (te - ts)
        print('%r  took %2.2f seconds' % (method.__name__, total_time))
        return result
    return timed
