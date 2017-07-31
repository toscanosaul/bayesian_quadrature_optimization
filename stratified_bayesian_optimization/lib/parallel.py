from __future__ import absolute_import

import multiprocessing as mp
import multiprocessing.pool
from multiprocessing.pool import ThreadPool


from stratified_bayesian_optimization.initializers.log import SBOLog

logger = SBOLog(__name__)


class Parallel(object):

    @classmethod
    def run_function_different_arguments_parallel(cls, function, arguments, all_success=False,
                                                  signal=None, parallel=True, threads=0,
                                                  *args, **kwargs):
        """
        Call functions in parallel
        :param function: f(argument, **kwargs)
        :param arguments: {i: argument}
        :param all_success: (boolean) the function will raise an exception if one of the runs
            fail and all_success is True
        :param signal: (function) calls this function after generating the jobs. It's used to test
            KeyboardInterrupt, and the signal is a mock of KeyboardInterrupt.
        :param parallel: (boolean) The code is run in parallel only if it's True.
        :param threads: (int) Uses threads instead of processes if threads > 0
        :param args: additional arguments of function
        :param kwargs: additional arguments of function
        :return: {int: output of f(arguments[i])}
        """
        # Maybe later we enable this feature.
        #thread = False

        jobs = {}

        if not parallel:
            return cls.run_function_different_arguments_sequentially(function, arguments, *args,
                                                                     **kwargs)

        n_jobs = min(len(arguments), mp.cpu_count())

        if threads > 0:
            pool = ThreadPool(threads)
        else:
            pool = mp.Pool(processes=n_jobs)

        try:
            for key, argument in arguments.iteritems():
                job = pool.apply_async(function, args=(argument, ) + args, kwds=kwargs)
                jobs[key] = job
            pool.close()
            pool.join()
            if signal is not None:
                signal(1)
        except KeyboardInterrupt:
            logger.info("Ctrl+c received, terminating and joining pool.")
            pool.terminate()
            pool.join()
            return -1

        results = {}
        for key in arguments.keys():
            try:
                results[key] = jobs[key].get()
            except Exception as e:
                if all_success:
                    raise e
                else:
                    logger.info("job failed" )
        return results

    @staticmethod
    def run_function_different_arguments_sequentially(function, arguments, *args, **kwargs):
        """
        Call functions in parallel
        :param function: f(argument, **kwargs)
        :param arguments: {i: argument}
        :param args: additional arguments of function
        :param kwargs: additional arguments of function
        :return: {int: output of f(arguments[i])}
        """
        results = {}

        for key, argument in arguments.iteritems():
            args_ = (argument, ) + args
            kwds = kwargs
            results[key] = function(*args_, **kwds)
        return results


class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.

class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
