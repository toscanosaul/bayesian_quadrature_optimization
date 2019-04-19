import numpy as np
import scipy.stats

from openaerostruct.geometry.utils import generate_mesh

from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint

from openmdao.api import IndepVarComp, Problem, Group, SqliteRecorder
from openmdao.api import ScipyOptimizeDriver


grav_constant = 9.81
n_scenarios = 8


data_heights = {}
data_heights[7000] = {'g':9.785, 'ro': 0.5900, 'visc': 1.561*1e-5, 'speed':312.251}
data_heights[10000] = {'g':9.776, 'ro': 0.4135, 'visc': 1.458*1e-5, 'speed':299.469}
data_heights[20000] = {'g':9.745, 'ro': 0.08891, 'visc':1.422*1e-5, 'speed':294.999}
data_heights[30000] = {'g':9.715, 'ro': 0.01841, 'visc':1.475*1e-5, 'speed':304.839}
data_heights[40000] = {'g':9.684, 'ro': 0.003996, 'visc':1.601*1e-5, 'speed':323.991}
data_heights[50000] = {'g':9.654, 'ro': 0.001027, 'visc':1.704*1e-5, 'speed':334.417}

def weights_points(n=8, mean=0.84, std=0.0067):
    """
    Compute weights of different flight conditions, i.e. different Mach numbers.
    See:
    https://deepblue.lib.umich.edu/bitstream/handle/2027.42/140677/1.J052940.pdf?sequence=1&isAllowed=y

    :param n:
    :param mean:
    :param std:
    :return:
    """
    z = np.linspace(0, 1, n + 1)
    points = []

    b = z[1]
    r = scipy.stats.norm.ppf(b, mean, std)
    points.append(r)
    for j in range(1, n - 1):
        a = z[j]
        b = z[j + 1]
        r = scipy.stats.norm.ppf(b, mean, std)
        r0 = scipy.stats.norm.ppf(a, mean, std)
        h = (r + r0) * 0.5
        points.append(h)
    b = z[-2]
    r = scipy.stats.norm.ppf(b, mean, std)
    points.append(r)
    distance = z[1]
    return points, distance

points, weight = weights_points(n_scenarios)


def get_burn_flight_conditions(thickness_cp, twist_cp, points=points, weight=weight):
    sol = 0.0
    count = 0.0
    for match_number in points:
        for height in data_heights:
            ans = get_burn(thickness_cp, twist_cp, match_number, height)
            sol += ans
            count += 1

    return sol / float(count)


def get_burn(thickness_cp, twist_cp, match_number, height):
    surface = get_dict(thickness_cp, twist_cp)
    prob = get_problem(surface, match_number, height)
    prob.run_driver()

    return prob['AS_point_0.fuelburn'][0]


def get_dict(thickness_cp, twist_cp_):

    # Create a dictionary to store options about the surface
    mesh_dict = {'num_y' : 5,
                 'num_x' : 2,
                 'wing_type' : 'CRM',
                 'symmetry' : True,
                 'num_twist_cp' : 5}

    mesh, twist_cp = generate_mesh(mesh_dict)

    surface = {
                # Wing definition
                'name' : 'wing',        # name of the surface
                'symmetry' : True,     # if true, model one half of wing
                                        # reflected across the plane y = 0
                'S_ref_type' : 'wetted', # how we compute the wing area,
                                         # can be 'wetted' or 'projected'
                'fem_model_type' : 'tube',
                'thickness_cp' : thickness_cp,

              #  'thickness_cp' : np.array([.1, .2, .3]),

               # 'twist_cp' : twist_cp,
                'twist_cp' : twist_cp_,
                'mesh' : mesh,

                # Aerodynamic performance of the lifting surface at
                # an angle of attack of 0 (alpha=0).
                # These CL0 and CD0 values are added to the CL and CD
                # obtained from aerodynamic analysis of the surface to get
                # the total CL and CD.
                # These CL0 and CD0 values do not vary wrt alpha.
                'CL0' : 0.0,            # CL of the surface at alpha=0
                'CD0' : 0.015,            # CD of the surface at alpha=0

                # Airfoil properties for viscous drag calculation
                'k_lam' : 0.05,         # percentage of chord with laminar
                                        # flow, used for viscous drag
                't_over_c_cp' : np.array([0.15]),      # thickness over chord ratio (NACA0015)
                'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
                                        # thickness
                'with_viscous' : True,
                'with_wave' : False,     # if true, compute wave drag

                # Structural values are based on aluminum 7075
                'E' : 70.e9,            # [Pa] Young's modulus of the spar
                'G' : 30.e9,            # [Pa] shear modulus of the spar
                'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
                'mrho' : 3.e3,          # [kg/m^3] material density
                'fem_origin' : 0.35,    # normalized chordwise location of the spar
                'wing_weight_ratio' : 2.,
                'struct_weight_relief' : False,    # True to add the weight of the structure to the loads on the structure
                'distributed_fuel_weight' : False,
                # Constraints
                'exact_failure_constraint' : False, # if false, use KS function
                }
    return surface


def get_problem(surface, match_number=0.84, height=7000):
    # Create the problem and assign the model group
    prob = Problem()

    grav_constant = data_heights[height]['g']
    v = match_number * data_heights[height]['speed']
    re = data_heights[height]['ro'] * data_heights[height]['speed'] * match_number / data_heights[height]['visc']
    rho = data_heights[height]['ro']
    speed_of_sound = data_heights[height]['speed']

    indep_var_comp = IndepVarComp()
    indep_var_comp.add_output('v', val=v, units='m/s')  # change this too
    indep_var_comp.add_output('alpha', val=5., units='deg')
    indep_var_comp.add_output('Mach_number', val=match_number)
    indep_var_comp.add_output('re', val=re, units='1/m')  # change this too
    indep_var_comp.add_output('rho', val=rho, units='kg/m**3')
    indep_var_comp.add_output('CT', val=grav_constant * 17.e-6, units='1/s')
    indep_var_comp.add_output('R', val=11.165e6, units='m')
    indep_var_comp.add_output('W0', val=0.4 * 3e5, units='kg')
    indep_var_comp.add_output('speed_of_sound', val=speed_of_sound, units='m/s')
    indep_var_comp.add_output('load_factor', val=1.)
    indep_var_comp.add_output('empty_cg', val=np.zeros((3)), units='m')

    prob.model.add_subsystem('prob_vars',
                             indep_var_comp,
                             promotes=['*'])

    aerostruct_group = AerostructGeometry(surface=surface)

    name = 'wing'

    # Add tmp_group to the problem with the name of the surface.
    prob.model.add_subsystem(name, aerostruct_group)

    point_name = 'AS_point_0'

    # Create the aero point group and add it to the model
    AS_point = AerostructPoint(surfaces=[surface])

    prob.model.add_subsystem(point_name, AS_point,
                             promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho', 'CT', 'R',
                                              'W0', 'speed_of_sound', 'empty_cg', 'load_factor'])

    com_name = point_name + '.' + name + '_perf'
    prob.model.connect(name + '.local_stiff_transformed',
                       point_name + '.coupled.' + name + '.local_stiff_transformed')
    prob.model.connect(name + '.nodes', point_name + '.coupled.' + name + '.nodes')

    # Connect aerodyamic mesh to coupled group mesh
    prob.model.connect(name + '.mesh', point_name + '.coupled.' + name + '.mesh')

    # Connect performance calculation variables
    prob.model.connect(name + '.radius', com_name + '.radius')
    prob.model.connect(name + '.thickness', com_name + '.thickness')
    prob.model.connect(name + '.nodes', com_name + '.nodes')
    prob.model.connect(name + '.cg_location',
                       point_name + '.' + 'total_perf.' + name + '_cg_location')
    prob.model.connect(name + '.structural_mass',
                       point_name + '.' + 'total_perf.' + name + '_structural_mass')
    prob.model.connect(name + '.t_over_c', com_name + '.t_over_c')

    prob.driver = ScipyOptimizeDriver()
    prob.driver.options['tol'] = 1e-9

    recorder = SqliteRecorder("aerostruct.db")
    prob.driver.add_recorder(recorder)
    prob.driver.recording_options['record_derivatives'] = True
    prob.driver.recording_options['includes'] = ['*']

    # Setup problem and add design variables, constraint, and objective
    # prob.model.add_design_var('wing.twist_cp', lower=-10., upper=15.)
    # prob.model.add_design_var('wing.thickness_cp', lower=0.01, upper=0.5, scaler=1e2)
    prob.model.add_constraint('AS_point_0.wing_perf.failure', upper=0.)
    prob.model.add_constraint('AS_point_0.wing_perf.thickness_intersects', upper=0.)

    # Add design variables, constraisnt, and objective on the problem
    prob.model.add_design_var('alpha', lower=-10., upper=10.)
    prob.model.add_constraint('AS_point_0.L_equals_W', equals=0.)
    prob.model.add_objective('AS_point_0.fuelburn', scaler=1e-5)
    prob.setup(check=True)
    return prob
