from __future__ import absolute_import

from scipy.optimize import minimize

def newton_cg(f, start, fprime, hessian, args, bounds, tol=None, **optimization_options):
    sol = minimize(f, start, args=args, method='Newton-CG', jac=fprime, hess=hessian,
                   options=optimization_options)
    new_solution = []

    point = sol['x']

    for dim, bound in enumerate(bounds):
        if bound[0] is not None:
            point[dim] = max(bound[0], point[dim])
        if bound[1] is not None:
            point[dim] = min(bound[1], point[dim])
    new_solution.append(point)
    new_solution.append(sol['fun'])

    res = {}
    res['grad'] = sol['jac']
    res['warnflag'] = sol['message']
    res['nit'] = sol['nit']
    res['funcalls'] = sol['nfev']
    res['task'] = sol['success']

    new_solution.append(res)

    return new_solution

def trust_ncg(f, start, fprime, hessian, args, bounds, tol=None, **optimization_options):
    sol = minimize(f, start, args=args, method='trust-ncg', jac=fprime, hess=hessian,
                   options=optimization_options)
    new_solution = []

    point = sol['x']

    for dim, bound in enumerate(bounds):
        if bound[0] is not None:
            point[dim] = max(bound[0], point[dim])
        if bound[1] is not None:
            point[dim] = min(bound[1], point[dim])
    new_solution.append(point)
    new_solution.append(sol['fun'])

    res = {}
    res['grad'] = sol['jac']
    res['warnflag'] = sol['message']
    res['nit'] = sol['nit']
    res['funcalls'] = sol['nfev']
    res['task'] = sol['success']

    new_solution.append(res)

    return new_solution

def dogleg(f, start, fprime, hessian, args, bounds, tol=None, **optimization_options):
    sol = minimize(f, start, args=args, method='dogleg', jac=fprime, hess=hessian, tol=tol,
                   options=optimization_options)
    new_solution = []

    point = sol['x']

    for dim, bound in enumerate(bounds):
        if bound[0] is not None:
            point[dim] = max(bound[0], point[dim])
        if bound[1] is not None:
            point[dim] = min(bound[1], point[dim])
    new_solution.append(point)
    new_solution.append(sol['fun'])

    res = {}
    res['grad'] = sol['jac']
    res['warnflag'] = sol['message']
    res['nit'] = sol['nit']
    res['funcalls'] = sol['nfev']
    res['task'] = sol['success']

    new_solution.append(res)

    return new_solution

