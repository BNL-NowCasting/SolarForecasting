import multiprocessing
import ctypes
import numpy as np
import mymodule
from time import sleep

lock = None

def init(shared_array_base):
    global lock
#     shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    mymodule.toShare = np.frombuffer(shared_array_base.get_obj())
    mymodule.toShare = mymodule.toShare.reshape(10, 10)
    print ('Init:',mymodule.toShare)
    lock = multiprocessing.Lock()

# Parallel processing
def my_func(i, def_param=(lock, )):
    lock.acquire() 
    print(i)
    mymodule.toShare[i, :] = i
#     print(mymodule.toShare)
    lock.release()    

if __name__ == '__main__':
    shared_array_base = multiprocessing.Array(ctypes.c_double, 10*10)
    
    pool = multiprocessing.Pool(processes=2, initializer=init, initargs=(shared_array_base,))
    pool.map(my_func, range(10))

# #     shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
#     shared_array = np.frombuffer(shared_array_base.get_obj())
#     shared_array = shared_array.reshape(10, 10)
#    sleep(2)
    print ('Final:',mymodule.toShare)


# from multiprocessing import Pool, Array, Process
# import mymodule
# 
# def count_it( key ):
#   count = 0
#   for c in mymodule.toShare:
#     if c == key.encode('utf-8'):
#       count += 1
#       
#   return count
# 
# def initProcess(share):
#   mymodule.toShare = share
# 
# if __name__ == '__main__':
#   # allocate shared array - want lock=False in this case since we 
#   # aren't writing to it and want to allow multiple processes to access
#   # at the same time - I think with lock=True there would be little or 
#   # no speedup
#   maxLength = 50
#   toShare = Array('c', maxLength, lock=False)
# 
#   # fork
#   pool = Pool(initializer=initProcess,initargs=(toShare,))
# 
#   # can set data after fork
#   testData = b"abcabcs bsdfsdf gdfg dffdgdfg sdfsdfsd sdfdsfsdf"
#   if len(testData) > maxLength:
#       raise ValueError("Shared array too small to hold data")
#   toShare[:len(testData)] = testData
# 
#   print(pool.map( count_it, ["a", "b", "s", "d"] ))

# import multiprocessing
# import ctypes
# import numpy as np
# from time import sleep
# 
# def shared_array(shape):
#     """
#     Form a shared memory numpy array.
#     
#     http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing 
#     """
#     
#     shared_array_base = multiprocessing.Array(ctypes.c_double, shape[0]*shape[1])
#     shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
#     shared_array = shared_array.reshape(*shape)
#     return shared_array
# 
# 
# # Form a shared array and a lock, to protect access to shared memory.
# array = shared_array((10, 10))
# lock = multiprocessing.Lock()
# 
# 
# def parallel_function(i, def_param=(lock, array)):
#     """
#     Function that operates on shared memory.
#     """
#     
#     # Make sure your not modifying data when someone else is.
#     lock.acquire()    
#     
#     array[i, :] = i
#     
#     # Always release the lock!
#     lock.release()
# 
# if __name__ == "__main__":
#     pool = multiprocessing.Pool(processes=3)
#         
#     # Call the parallel function with different inputs.
#     args = [(0), 
#             (1), 
#             (2)]
#     
#     # Use map - blocks until all processes are done.
#     pool.map(parallel_function, args )
# 
#     sleep(2)
#     
#     print(array)
    
# import multiprocessing
# from multiprocessing import Manager
# import functools
# 
# def consume(ele, dic):
#     if dic.get(ele) is None:
#         dic[ele] = ele**2
#     return ele**2
# 
# if __name__ == '__main__': 
#     m_list = [9,8,7,6,5,10]
# # Use proc manager to create a shared memory dict across processes
#     proc_manager = Manager()
#     m_dict = proc_manager.dict([(s, s**2) for s in [3,4,5,6]])
#     pool = multiprocessing.Pool(processes=3)
#     result = pool.map(functools.partial(consume, dic=m_dict), m_list)
#     print (result)
#     print (m_dict)

# import multiprocessing, ctypes
# from time import sleep
# 
# count = multiprocessing.Value(ctypes.c_int, 0)  # (type, init value)
# 
# def smile_detection(thread_name, count):
# 
#     for x in range(10):
#         count.value +=1
#         print(thread_name,count.value)
# 
#     return count    
# 
# if __name__ == '__main__':
#     x = multiprocessing.Process(target=smile_detection, args=("Thread1", count))
#     y = multiprocessing.Process(target=smile_detection, args=("Thread2", count))
#     x.start()
#     y.start()
#     sleep(2)
#     print('count is:',count.value)
    
# import multiprocessing
# import ctypes
# import numpy as np
# 
# shared_array = None
# lock = None
# 
# def init(shared_array_base):
#     global shared_array,lock
# #     shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
#     shared_array = np.frombuffer(shared_array_base.get_obj())
#     shared_array = shared_array.reshape(10, 10)
#     lock = multiprocessing.Lock()
# 
# # Parallel processing
# def my_func(i, def_param=(lock, shared_array)):
#     lock.acquire() 
#     shared_array[i, :] = i
#     lock.release()    
# 
# if __name__ == '__main__':
#     shared_array_base = multiprocessing.Array(ctypes.c_double, 10*10)
#     
#     pool = multiprocessing.Pool(processes=4, initializer=init, initargs=(shared_array_base,))
#     pool.map(my_func, range(10))
# 
# # #     shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
#     shared_array = np.frombuffer(shared_array_base.get_obj())
#     shared_array = shared_array.reshape(10, 10)
#     print (shared_array)

# from multiprocessing import Process, Value, Array
# 
# def f(n, a):
#     n.value += 1
#     for i in range(len(a)):
#         a[i] = -a[i]       
# 
# if __name__ == '__main__':
#     num = Value('d', 0.0)
#     arr = Array('i', range(10))
# 
#     p = Process(target=f, args=(num, arr))
#     p.start()
#     p.join()
#     
#     p = Process(target=f, args=(num, arr))
#     p.start()
#     p.join()  
# 
# 
#     print(num.value)
#     print(arr[:])
