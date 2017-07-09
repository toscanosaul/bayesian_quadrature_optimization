from __future__ import absolute_import

import multiprocessing as mp

from stratified_bayesian_optimization.initializers.log import SBOLog

logger = SBOLog(__name__)


class Parallel(object):

    @staticmethod
    def run_function_different_arguments_parallel(function, arguments, all_success=False,
                                                  signal=None, **kwargs):
        """
        Call functions in parallel
        :param function: f(argument, **kwargs)
        :param arguments: {i: argument}
        :param all_success: (boolean) the function will raise an exception if one of the runs
            fail and all_success is True
        :param signal: (function) calls this function after generating the jobs. It's used to test
            KeyboardInterrupt, and the signal is a mock of KeyboardInterrupt.
        :param kwargs: additional arguments of function
        :return: {int: output of f(arguments[i])}
        """
        jobs = {}

        n_jobs = min(len(arguments), mp.cpu_count())

        pool = mp.Pool(processes=n_jobs)
        try:
            for key, argument in arguments.iteritems():
                job = pool.apply_async(function, args=(argument, ), kwds=kwargs)
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

        for key in arguments:
            try:
                results[key] = jobs[key].get()
            except Exception as e:
                if all_success:
                    raise e
                else:
                    logger.info("job %d failed" % key)
        return results
