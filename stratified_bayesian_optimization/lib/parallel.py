from __future__ import absolute_import

import multiprocessing as mp

from stratified_bayesian_optimization.initializers.log import SBOLog

logger = SBOLog(__name__)

class Parallel(object):

    @staticmethod
    def run_function_different_arguments_parallel(function, arguments, all_success=False, **kwargs):
        """
        Call functions in parallel
        :param function: f(argument, **kwargs)
        :param arguments: {i: argument}
        :param all_success: (boolean) the function will raise an exception if one of the runs
            fail and all_success is True
        :param **kwargs
        :return: {int: output of f(arguments[i])}
        """
        jobs = {}

        n_jobs = min(len(arguments), mp.cpu_count())

        try:
            pool = mp.Pool(processes=n_jobs)
            for key, argument in arguments.iteritems():
                job = pool.apply_async(function, args=(argument, ), kwds=kwargs)
                jobs[key] = job
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            logger.info("Ctrl+c received, terminating and joining pool.")
            pool.terminate()
            pool.join()

        results = {}

        for key in arguments:
            try:
                results[key] = jobs[key].get()
            except Exception as e:
                if all_success:
                    raise e
                else:
                    logger.info("job %d failed"%key)

        return results
