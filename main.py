from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import math

from scipy.integrate import odeint
from scipy.optimize import fsolve

from sympy import Matrix, Symbol, lambdify, re, solve, symbols, simplify
import copy

# class Bifurcation:
#     def __init__(self, f1: callable, f2: callable, variables: List[Symbol], param: Symbol):
#         self.result = solve([f1, f2], variables[0], param)[0]
#         self.A_matrix = Matrix([f1, f2])
#         self.var_vector = Matrix(variables)
#         self.jac_A = self.A_matrix.jacobian(self.var_vector)
#         self.det_A = self.jac_A.det()
#         self.trace_A = self.jac_A.trace()
#
#     def one_parameter_analysis_v1(
#         self,
#         f1: callable,
#         f2: callable,
#         constants_values: Dict[Symbol, float],
#         variables: List[Symbol],
#         param: Symbol,
#     ):
#         # get keys and appropriate values without the observed parameter
#         keys = list(constants_values.keys())
#         values = list(constants_values.values())
#         ind = keys.index(param)
#         keys.remove(param)
#         values.pop(ind)
#
#         # get k2(y) function
#         x_y = solve(f2, variables[0])[0]
#         f1_y_with_constants = f1.subs([(variables[0], x_y)])
#         k2_y = solve(f1_y_with_constants, param)[0]
#         k2_y = k2_y.subs([(key, value) for key, value in zip(keys, values)])
#
#         # get bifurcation points
#         det = self.det_A
#         det = det.subs([(variables[0], x_y)])
#         det = det.subs([(key, value) for key, value in zip(keys, values)])
#         trace = self.trace_A
#         trace = trace.subs([(variables[0], x_y)])
#         trace = trace.subs([(key, value) for key, value in zip(keys, values)])
#         trace_k2 = solve(trace, param)[0]
#         bifurcation_y = det.subs([(param, trace_k2)])
#         bifurcation_y = lambdify(variables[1], bifurcation_y, 'numpy')
#         y0 = np.array([0.22])
#         y_point = fsolve(bifurcation_y, x0=y0)[0]
#
#         # get x,y dependence from k2
#         y_arange = np.arange(0, 0.5, 0.005)
#         x_y = x_y.subs([(key, value) for key, value in zip(keys, values)])
#         x_y = lambdify(variables[1], x_y, 'numpy')
#         k2_y = lambdify(variables[1], k2_y, 'numpy')
#         x_arange = x_y(y_arange)
#         k2_arange = k2_y(y_arange)
#
#         # draw x,y dependence from k2
#         plt.plot(k2_arange, y_arange, linewidth=1.5, label='multi')
#         plt.plot(k2_arange, x_arange, linestyle='--', linewidth=1.5, label='neutral')
#
#         # draw bifurcation points
#         plt.plot(k2_y(y_point), y_point, 'ro')
#         plt.plot(k2_y(y_point), x_y(y_point), 'ro')
#
#         plt.xlabel(r'$k2$')
#         plt.ylabel(r'$x, y$')
#         plt.xlim([-3, 5])
#         plt.ylim([0, 1])
#         plt.show()
#
#     def one_parameter_analysis_v2(
#         self,
#         f1: callable,
#         f2: callable,
#         constants_values: Dict[Symbol, float],
#         variables: List[Symbol],
#         param: Symbol,
#     ):
#         # get keys and appropriate values without the observed parameter
#         keys = list(constants_values.keys())
#         values = list(constants_values.values())
#         ind = keys.index(param)
#         keys.remove(param)
#         values.pop(ind)
#
#         eq1 = f1.subs([(key, value) for key, value in zip(keys, values)])
#         eq2 = f2.subs([(key, value) for key, value in zip(keys, values)])
#
#         sol = solve([eq1, eq2], variables[0], param, dict=True)[0]
#         y_lin = np.linspace(0.001, 0.5, 500, endpoint=True)
#
#         k2_array = np.array([sol[param].subs({variables[1]: i}) for i in y_lin])
#         x_array = np.array([sol[variables[0]].subs({variables[1]: i}) for i in y_lin])
#
#         jac_A = self.jac_A.subs([(key, value) for key, value in zip(keys, values)])
#         eig = list(jac_A.eigenvals().keys())
#         dd = []
#         for i in range(len(eig)):
#             eig_eval = eig[i].subs({param: sol[param], variables[0]: sol[variables[0]]})
#             y_lambda = solve(eig_eval, variables[1])
#             dd.extend(list(filter(lambda lam: lam >= 0, map(re, y_lambda))))
#
#         feature_y = dd
#         feature_k1 = [sol[param].subs({variables[1]: i}) for i in feature_y]
#         feature_x = [sol[variables[0]].subs({variables[1]: i}) for i in feature_y]
#
#         sn1_abs = [feature_k1[0], feature_k1[0]]
#         sn1_ord = [feature_y[0], feature_x[0]]
#         sn2_abs = [feature_k1[1], feature_k1[1]]
#         sn2_ord = [feature_y[1], feature_x[1]]
#
#         # draw x,y dependence from k2
#         plt.plot(k2_array, y_lin, linewidth=1.5, label='multi')
#         plt.plot(k2_array, x_array, linestyle='--', linewidth=1.5, label='neutral')
#
#         # draw bifurcation points
#         plt.plot(sn1_abs, sn1_ord, 'ro')
#         plt.plot(sn2_abs, sn2_ord, 'X')
#
#         plt.xlabel(r'$k2$')
#         plt.ylabel(r'$x, y$')
#         plt.xlim([-3, 5])
#         plt.ylim([0, 1])
#         plt.show()
#
#     def two_parameter_analysis(
#         self,
#         f1: callable,
#         f2: callable,
#         constants_values: Dict[Symbol, float],
#         variables: List[Symbol],
#         params: List[Symbol],
#     ):
#         # get keys and appropriate values without the observed parameters
#         keys = list(constants_values.keys())
#         values = list(constants_values.values())
#         for param in params:
#             ind = keys.index(param)
#             keys.remove(param)
#             values.pop(ind)
#
#         # get k2(x) function
#         y_x = solve(f2, variables[1])[0]
#         f1 = f1.subs([(variables[1], y_x)])
#         k2_x = solve(f1, params[1])[0]
#
#         multiplicity_line = self.det_A
#         multiplicity_line = multiplicity_line.subs([(variables[1], y_x)])
#         k2_multiplicity = solve(multiplicity_line, params[1])[0]
#
#         neutrality_line = self.trace_A
#         neutrality_line = neutrality_line.subs([(variables[1], y_x)])
#         k2_neutrality = solve(neutrality_line, params[1])[0]
#
#         k_1_multiplicity = solve(k2_multiplicity - k2_x, params[0])[0]
#         k_1_neutrality = solve(k2_neutrality - k2_x, params[0])[0]
#
#         k_1_multiplicity = k_1_multiplicity.subs([(key, value) for key, value in zip(keys, values)])
#         k_1_neutrality = k_1_neutrality.subs([(key, value) for key, value in zip(keys, values)])
#         k2_x = k2_x.subs([(key, value) for key, value in zip(keys, values)])
#
#         k2_multiplicity = k2_x.subs([(params[0], k_1_multiplicity)])
#         k2_neutrality = k2_x.subs([(params[0], k_1_neutrality)])
#         k2_multiplicity = lambdify(variables[0], k2_multiplicity, 'numpy')
#         k2_neutrality = lambdify(variables[0], k2_neutrality, 'numpy')
#
#         k_1_multiplicity = lambdify(variables[0], k_1_multiplicity, 'numpy')
#         k_1_neutrality = lambdify(variables[0], k_1_neutrality, 'numpy')
#         x_arange = np.arange(0, 1, 0.05)
#         k_1_multiplicity_arange = k_1_multiplicity(x_arange)
#         k_1_neutrality_arange = k_1_neutrality(x_arange)
#         k2_multiplicity_arange = k2_multiplicity(x_arange)
#         k2_neutrality_arange = k2_neutrality(x_arange)
#         # bogdanov_takens = self.jac_A
#         # bogdanov_takens = bogdanov_takens.subs([(params[1], k2_x)])
#         # bogdanov_takens = bogdanov_takens.subs([(key, value) for key, value in zip(keys, values)])
#         # bogdanov_takens = bogdanov_takens.subs([(variables[0], self.result[0]), (params[1], self.result[1])])
#         # eigenvalues = bogdanov_taken.eigenvals()
#         # print(eigenvalues.keys())
#         # print(eigenvalues.values())
#         # bogdanov_takens = solve(eigenvalues.keys(), params[0], params[1])
#         plt.plot(k_1_multiplicity_arange, k2_multiplicity_arange, linewidth=1.5, label='multi')
#         plt.plot(k_1_neutrality_arange, k2_neutrality_arange, linestyle='--', linewidth=1.5, label='neutral')
#         plt.xlabel(r'$k1$')
#         plt.ylabel(r'$k2$')
#         plt.xlim([0, 0.5])
#         plt.ylim([0, 3])
#         plt.show()

def draw(tab):
    fig, ax = plt.subplots()

    # Сплошная линия ('-' или 'solid',
    # установлен по умолчанию):
    ax.plot(tab['param'], tab['x'],
                linestyle='-',
                linewidth=1,
                color='red', label=r'x')

    ax.plot(tab['param'], tab['y'],
            linestyle='-',
            linewidth=1,
            color='green', label=r'y')

    for p in tab['biff_points']:
        ax.plot(p[0], p[1], 'ro')

    ax.legend()
    fig.set_figwidth(12)
    fig.set_figheight(6)
    fig.set_facecolor('linen')
    ax.set_facecolor('ivory')
    ax.set_xlabel('k2')
    ax.set_ylabel('x,y')

    plt.xlim([-1, 8])
    plt.show()

def draw2(x, y, u, v, x_lims = None, y_lims= None):
    fig, ax = plt.subplots()
    colors = ['green', 'red', 'black', 'blue', 'gray', 'yellow']
    linestyles = ['-', '--', 'ro', 'X']
    for i, x_ in enumerate(x):
        ax.plot(x_, y[i],
                linestyle=linestyles[i],
                linewidth=1,
                color=colors[i])
    for j, u_ in enumerate(u):
        ax.plot(u_, v[j], linestyles[j+2])
        # , label=r'y')


    # for p in tab['biff_points']:
    #     ax.plot(p[0], p[1], 'ro')

    ax.legend()
    fig.set_figwidth(12)
    fig.set_figheight(6)
    fig.set_facecolor('linen')
    ax.set_facecolor('ivory')
    ax.set_xlabel(r'$k_{1}$')
    ax.set_ylabel(r'$k_{-1}$')
    if x_lims is None:
        pass
    else:
        plt.xlim(x_lims)
    if y_lims is None:
        pass
    else:
        plt.ylim(y_lims)
    plt.show()

def draw3(xy, x_lim, y_lim, leg1, lab1):
    fig, ax = plt.subplots()
    linestyles = ['-','--','--']
    labels = ['x', 'y']
    # Сплошная линия ('-' или 'solid',
    # установлен по умолчанию):

    for i, xy_ in enumerate(xy):
        ax.plot(xy_[0], xy_[1], linestyle=linestyles[i], linewidth=1.5)


    ax.legend()
    fig.set_figwidth(12)
    fig.set_figheight(6)
    fig.set_facecolor('linen')
    ax.set_facecolor('ivory')
    ax.set_xlabel('t')
    ax.set_ylabel(leg1)

    if x_lim is None:
        pass
    else:
        plt.xlim(x_lim)
    if y_lim is None:
        pass
    else:
        plt.ylim(y_lim)
    plt.show()

def draw4(xy, sol_cycle, x_lim, y_lim):
    fig, ax = plt.subplots()
    linestyles = ['-','-','--']
    linestyles2 = ['-', '-.', '--', ':']
    colors = ['orange','orange','blue', ]
    colors2 = ['black', 'red', 'blue', 'green']

    for i, xy_ in enumerate(xy):
        ax.plot(xy_[0], xy_[1], linestyle=linestyles[i], linewidth=1.5, color = colors[i], label='f1(x,y)=0')

    for j, s in enumerate(sol_cycle):
            if j==len(sol_cycle)-1:
            # ax.plot(s[:, 0], s[:, 1], linestyle=linestyles2[j%4], linewidth=1.5,
            #             color=colors2[j%4], label='f1(x,y)=0')
                ax.quiver(
                    s[:-1, 0],
                    s[:-1, 1],
                    s[1:, 0] - s[:-1, 0],
                    s[1:, 1] - s[:-1, 1],
                    scale=2,color='black',
                    label='cycle',
                )
            else:
                ax.plot(s[:, 0], s[:, 1], linestyle='-', linewidth=0.5,
                                    color='gray', label=f'trajectory{j+1}')
    # ax.quiver(
    #     sol_cycle[:-1, 0],
    #     sol_cycle[:-1, 1],
    #     sol_cycle[1:, 0] - sol_cycle[:-1, 0],
    #     sol_cycle[1:, 1] - sol_cycle[:-1, 1],
    #     scale=2,
    #     label='cycle',
    # )
    fig.set_figwidth(12)
    fig.set_figheight(6)
    fig.set_facecolor('linen')
    ax.set_facecolor('ivory')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if x_lim is None:
        pass
    else:
        plt.xlim(x_lim)
    if y_lim is None:
        pass
    else:
        plt.ylim(y_lim)
    plt.show()

def one_param_analisys(y_i, param):
    table = {'y':[], 'x': [], 'param': [], 'det': [], 'trace': [], 'la_i': {}, 'biff_points':[]}

    subs_dict = {p: exact_values[p] for p in exact_values if p is not param}

    solved =solve([y_ for y_ in y_i], x, param, dict=True)[0]

    x_i_y = solved[x]
    k_j_y = solved[param]
    y_pre_values = np.arange(0.001,1,0.001)
    x_i_y = x_i_y.subs(subs_dict)
    k_j_y = k_j_y.subs(subs_dict)
    x_i_y_func = lambdify(y, x_i_y)
    k_j_y_func = lambdify(y, k_j_y)

    x_values = []
    y_values = []
    for y_ in y_pre_values:
        x_ = x_i_y_func(y_)
        if x_>=0 and x_<=1 and x_+y_>=0 and x_+y_<=1:
            x_values.append(x_)
            y_values.append(y_)

    x_values = np.array(x_values)
    y_values = np.array(y_values)
    k_values = k_j_y_func(y_values)

    table['x'] = x_values
    table['y'] = y_values
    table['param'] = k_values

    A = Matrix(y_i)
    var_vector = Matrix([x,y])
    jacA = A.jacobian(var_vector)
    det_jacA = jacA.det()
    trace_jacA = jacA.trace()
    la = symbols('la')
    laE = Matrix([[la,0],[0,la]])
    jacA_laE = jacA-laE
    det_jacA_laE = jacA_laE.det()

    la_solved = solve(det_jacA_laE, la)
    la_solved_funcs = []
    for l in la_solved:
        l = l.subs(subs_dict)
        la_solved_funcs.append(lambdify([x,y,param], l))


    for i, lf in enumerate(la_solved_funcs):
        table['la_i'][f'la_{i}']=lf(table['x'], table['y'], table['param'])

    for laaa in la_solved:
        print(laaa)
    for i, y_ in enumerate(table['y']):
        if i > 0:
            for key, value in table['la_i'].items():
                if value[i-1].real*value[i].real < 0:
                    print('value: ', value[i])
                    y_biff = table['y'][i-1] - value[i-1].real*(y_-table['y'][i-1])/(value[i].real-value[i-1].real)
                    x_biff = table['x'][i-1] - value[i-1].real*(table['x'][i]-table['x'][i-1])/(value[i].real-value[i-1].real)
                    k_biff = table['param'][i-1] - value[i-1].real*(table['param'][i]-table['param'][i-1])/(value[i].real-value[i-1].real)
                    table['biff_points'].append((k_biff, y_biff))
                    table['biff_points'].append((k_biff, x_biff))

    print(table['biff_points'])
    draw(table)
    return table

def two_params_analisys(f, variables, params, tab):
    table = {'y':[], 'x': [], 'param': [], 'det': [], 'trace': [], 'la_i': {}, 'biff_points':[]}

    subs_dict = {p:exact_values[p] for p in exact_values if p not in params}
    print('subs_dict')
    # print(subs_dict)
    # print(f[0])
    # print(f[1])
    # solved = solve(f, variables[0], params[0], dict=True)[0]

    var2_pre_values = np.arange(0, 1, 0.0001)

    # solved_var1 = solved[variables[0]]
    # solved_param0 = solved[params[0]]

    # solved_var1, solved_param0 = solve(f, variables[1], params[0])[0]
    solved_var1 = solve(f[1], variables[1])[0]
    solved_param0 = solve(f[0], params[0])[0]

    print('solved_var1', solved_var1)
    print('solved_param0', solved_param0)

    solved_param0 = solved_param0.subs(variables[1], solved_var1)

    A = Matrix(f)
    var_vector = Matrix(variables)
    jacA = A.jacobian(var_vector)
    det_jacA = jacA.det()
    trace_jacA = jacA.trace()

    det_jacA = det_jacA.subs(variables[1], solved_var1)
    trace_jacA = trace_jacA.subs(variables[1], solved_var1)

    solved_param0_det = solve(det_jacA, params[0], dict = True)[0][params[0]]

    print('solved_param0_det', solved_param0_det)
    solved_param0_trace = solve(trace_jacA, params[0], dict=True)[0][params[0]]
    print('solved_param0_trace', solved_param0_trace)
    solved_param1_det = solve(solved_param0_det-solved_param0, params[1], dict = True)[0][params[1]]
    solved_param1_trace = solve(solved_param0_trace-solved_param0, params[1], dict=True)[0][params[1]]
    print('solved_param1_det', solved_param1_det)
    print('solved_param1_trace', solved_param1_trace)
    print('solved_var1.subs(subs_dict)', solved_var1.subs(subs_dict))
    solved_var1_func = lambdify(variables[0], solved_var1.subs(subs_dict))
    var2_values = np.array([val for val in var2_pre_values if val>=0 and val<=1 and
                   solved_var1_func(val)>=0 and solved_var1_func(val)<=1 and
                   solved_var1_func(val)+val>=0 and solved_var1_func(val)+val<=1])
    # var2_values = var2_pre_values
    var1_values = solved_var1_func(var2_values)
    print('var2_values', var2_values)
    solved_param1_det_func = lambdify(variables[0], solved_param1_det.subs(subs_dict))
    solved_param1_trace_func = lambdify(variables[0], solved_param1_trace.subs(subs_dict))

    param1_det_values = solved_param1_det_func(var2_values)
    param1_trace_values = solved_param1_trace_func(var2_values)

    solved_param0_det_func = lambdify((variables[0],params[1]), solved_param0_det.subs(subs_dict))
    solved_param0_trace_func = lambdify((variables[0],params[1]), solved_param0_trace.subs(subs_dict))

    param0_det_values = solved_param0_det_func(var2_values, param1_det_values)
    param0_trace_values = solved_param0_trace_func(var2_values, param1_trace_values)
    k1_val = []
    k2_val = []
    jacA = A.jacobian(var_vector)
    draw2(param0_det_values, param0_trace_values, param1_det_values, param1_trace_values, [], [], [], [])
    laE = Matrix(variables)
    jacA = jacA.subs(variables[1], solved_var1)
    jacA = jacA.subs(params[0], solved_param0)
    jacA = jacA.subs(subs_dict)
    print('jacA', jacA)
    eig = list(jacA.eigenvals().keys())
    print('eig', eig)
    aaa = solve(eig, params[1], dict=True)
    # print('aaa', aaa)
    print('eig[0]', eig[0])
    print('eig[1]', eig[1])
    eig[0] = lambdify((variables[0], params[1]), eig[0])
    eig[1] = lambdify((variables[0], params[1]), eig[1])
    x_val = []
    for x_ in var2_values:
        if eig[0](x_, solved_param1_det_func(x_))<=0.00001 and eig[1](x_, solved_param1_det_func(x_))<=0.00001 and \
                eig[0](x_, solved_param1_trace_func(x_))<=0.00001 and eig[1](x_, solved_param1_trace_func(x_))<=0.00001:
            x_val.append(x_)
            # if x_val is None:
            #     x_val = x_
            # else:
            #     if eig[0](x_, solved_param1_det_func(x_))<=eig[0](x_val, solved_param1_det_func(x_val)) and \
            #             eig[1](x_, solved_param1_det_func(x_))<=eig[1](x_val, solved_param1_det_func(x_val)) and \
            #     eig[0](x_, solved_param1_trace_func(x_))<=eig[0](x_val, solved_param1_trace_func(x_val)) and \
            #             eig[1](x_, solved_param1_trace_func(x_))<=eig[1](x_val, solved_param1_trace_func(x_val)):
            #         x_val = x_
        # if :
        #     x_val.append(x_)

    # x_val = [x_val, ]
    print('solved_param0.subs(subs_dict)', solved_param0.subs(subs_dict))
    param0_func = lambdify((variables[0],params[1]), solved_param0.subs(subs_dict))
    k1_val = [param0_func(x_, solved_param1_det_func(x_)) for x_ in x_val]
    k1_val.extend([param0_func(x_, solved_param1_trace_func(x_)) for x_ in x_val])

    k2_val = [solved_param1_det_func(x_) for x_ in x_val]
    k2_val.extend([solved_param1_trace_func(x_) for x_ in x_val])

    print('k1_val', k1_val)
    print('k2_val', k2_val)

    print('det_jacA', det_jacA)
    diff_det_jacA = det_jacA.diff(variables[0])
    print('diff_det_jacA', diff_det_jacA)
    diff_det_jacA = diff_det_jacA.subs(subs_dict)
    print('diff_det_jacA', diff_det_jacA)
    x_c = solve(diff_det_jacA, variables[0])
    print('x_c', x_c)

    # f_diff = f[0].diff(variables[0]), f[1].diff(variables[1])
    # f_diff2 = f_diff[0].diff(variables[0]), f_diff[1].diff(variables[1])
    #
    # yyy = solve([f[0], f[1], f_diff[0], f_diff[1], f_diff2[0], f_diff2[1]], variables[0], variables[1], dict= True)
    # print(yyy)

    # print('x_c', type(x_c[0]), x_c)
    x_val2 = []

    for x_ in x_c:
        x_c_ = solve(x_, params[1])
        print('x_c_', x_c_)
        for x__ in x_c_:
            x_val2.append(x__)
        # print('x_c_', x_c_)
        # for x__ in x_c_:
        #     x_c_c = solve(x__, params[1])
        #     print('x_c_c', x_c_c)
        #     x_val2.extend(x_c_c)
    print('x_val2', x_val2)
    # x_c = lambdify((params[0], params[1]), x_c)

    # for x_ in var2_values:
    #     # print(x_c(param0_func(x_, solved_param1_det_func(x_)), solved_param1_det_func(x_)))
    #     if math.fabs(x_c(param0_func(x_, solved_param1_det_func(x_)), solved_param1_det_func(x_)))<=0.0001:
    #         # if x_val2 is None:
    #         #     x_val2 = x_
    #         x_val2.append(x_)
    #         # else:
    #         #     if x_c(param0_func(x_, solved_param1_det_func(x_)), solved_param1_det_func(x_))<=\
    #         #             x_c(param0_func(x_val2, solved_param1_det_func(x_val2)), solved_param1_det_func(x_val2)):
    #         #         x_val2 = x_
    # k1_val.append(param0_func(x_c, solved_param1_det_func(x_c)))
    # k2_val.append(solved_param1_det_func(x_c))
    # x_val2 = [x_val2, ]
    k1_val2=[param0_func(x_, solved_param1_det_func(x_)) for x_ in x_val2]
    print(solved_param1_det_func(x_val2[0]))
    k2_val2=[solved_param1_det_func(x_) for x_ in x_val2]
    #
    #
    draw2(param0_det_values,param0_trace_values, param1_det_values, param1_trace_values, k1_val, k2_val, k1_val2, k2_val2)
    draw2(param0_det_values, [], param1_det_values, [], k2_val, k1_val, k1_val2, k2_val2)
    draw2([],param0_trace_values, [], param1_trace_values, k1_val, k2_val, k1_val2, k2_val2)


def two_params_analisys2(f, variables, params):
    subs_dict = {p:exact_values[p] for p in exact_values if p not in params}
    solved_f = solve(f, variables[0], params[0], dict=True)

    print(len(solved_f), solved_f)
    for solution in solved_f:
        var1 = solution[variables[0]]
        par1 = solution[params[0]]

        # var1 = solution[variables[0]].subs(subs_dict)
        # par1 = solution[params[0]].subs(subs_dict)
        var1_func = lambdify(variables[1], var1.subs(subs_dict))


        print('var1', var1)
        print('par1', par1)

        var2_pre_values = np.arange(0.0001,0.9999,0.00001)
        var2_values = []
        for value in var2_pre_values:
            if var1_func(value)<=1 and var1_func(value)>=0 and var1_func(value)+value<=1 and var1_func(value)+value>=0:
                var2_values.append(value)
        if len(var2_values)<=1:
            print('Решения не существет для:')
            print('var1', var1)
            print('par2', par1)
            continue
        var2_values = np.array(var2_values)
        # var2_values = np.array(var2_pre_values)
        print('var2_values', var2_values)
        var1_values = var1_func(var2_values)
        print('var1_values', var1_values)

        A = Matrix(f)
        var_vector = Matrix([x,y])

        jacA = A.jacobian(var_vector)
        det_jacA = jacA.det()
        trace_jacA = jacA.trace()

        # jacA = jacA.subs(subs_dict)
        # det_jacA = det_jacA.subs(subs_dict)
        # trace_jacA = trace_jacA.subs(subs_dict)

        par1_det = solve(det_jacA.subs(variables[0], var1), params[0], dict=True)
        print('len(par1_det)', len(par1_det), par1_det)
        par1_det = par1_det[0][params[0]]

        par1_trace = solve(trace_jacA.subs(variables[0], var1), params[0], dict=True)
        print('len(par1_trace)', len(par1_trace), par1_trace)
        par1_trace = par1_trace[0][params[0]]

        par2_det = solve(par1_det-par1, params[1], dict=True)
        print('len(par2_det)', len(par2_det), par2_det)
        par2_det = par2_det[0][params[1]]

        par2_trace = solve(par1_trace-par1, params[1], dict=True)
        print('len(par2_trace)', len(par2_trace), par2_trace)
        par2_trace = par2_trace[0][params[1]]

        par2_det_func = lambdify(variables[1], par2_det.subs(subs_dict))
        par2_det_values = par2_det_func(var2_values)

        par2_trace_func = lambdify(variables[1], par2_trace.subs(subs_dict))
        par2_trace_values = par2_trace_func(var2_values)
        #
        par1_func = lambdify([variables[1], params[1]], par1.subs(subs_dict))
        par1_det_values = par1_func(var2_values, par2_det_values)
        par1_trace_values = par1_func(var2_values, par2_trace_values)

        print('started eig')
        eig = list(jacA.eigenvals().keys())
        eig[0] = eig[0].subs(variables[0], var1).subs(params[0], par1).subs(subs_dict)
        eig[1] = eig[1].subs(variables[0], var1).subs(params[0], par1).subs(subs_dict)

        print('eig', eig)

        eig_funcs = lambdify((variables[1], params[1]), eig[0]), lambdify((variables[1], params[1]), eig[1])

        bt_var2_values = []

        for i, value in enumerate(var2_values):
            if eig_funcs[0](value, par2_det_func(value))<=0.0000001 and \
                    eig_funcs[0](value, par2_trace_func(value))<=0.0000001 and \
                    eig_funcs[1](value, par2_det_func(value))<=0.0000001 and \
                    eig_funcs[1](value, par2_trace_func(value))<=0.0000001:
                bt_var2_values.append(value)

        k2_values = [par2_trace_func(value) for value in bt_var2_values]
        k1_values = [par1_func(value, par2_trace_func(value)) for value in bt_var2_values]
        #

        print('det_jacA', det_jacA)
        diff_det_jacA = par2_det.diff(variables[1])

        print('diff_det_jacA', diff_det_jacA)
        diff_det_jacA = diff_det_jacA.subs(variables[0], var1).subs(subs_dict)
        print('diff_det_jacA', diff_det_jacA)
        x_c = solve(diff_det_jacA, variables[1])
        print('x_c', x_c)
        # k1_c = solve(x_c, params[0])[0]
        # print(k1_c)
        # k2_c = solve(k1_c, params[1])[0]
        # print(k2_c)

        # xxx = x_c.subs(params[0], k1_c).subs(params[1], k2_c)
        xxx= x_c[1]
        print(par1_func(xxx,par2_trace_func(xxx)),par1_func(xxx,par2_det_func(xxx)),par2_trace_func(xxx),par2_det_func(xxx))

        nearest_x = None
        vvv = max(par2_trace_values)
        for x_ in var2_values:
            if nearest_x is None:
                nearest_x = x_
            else:
                if math.fabs(par2_trace_func(x_)-vvv)<math.fabs(par2_trace_func(nearest_x)-vvv):
                    nearest_x = x_

        print(nearest_x, par2_trace_func(nearest_x), par1_func(nearest_x, par2_trace_func(nearest_x)))


        # draw2([par1_det_values, par1_trace_values], [par2_det_values, par2_trace_values], [k1_values,], [k2_values,], [0, 0.4], [0, 0.05])
        draw2([par1_det_values, par1_trace_values], [par2_det_values, par2_trace_values], [k1_values, [par1_func(xxx,par2_det_func(xxx)),],], [k2_values,[par2_det_func(xxx),]], None, [-0.05, 0.05])
        return (nearest_x, par2_trace_func(nearest_x), par1_func(nearest_x, par2_trace_func(nearest_x)))

def auto(f, variables, params, params_value):
    subs_dict = {p: exact_values[p] for p in exact_values if p not in params}
    iterations = 1e6
    dt = 1e-2

    funcs = [lambdify(variables, f_.subs(subs_dict).subs(params[0], params_value[0]).subs(params[1], params_value[1])) for f_ in f]

    times = np.arange(iterations) * dt

    def oscillation_internal(vars, t):
        return [funcs[0](vars[0], vars[1]), funcs[1](vars[0], vars[1])]

    solution = odeint(oscillation_internal, y0=[0.5,0.25], t=times)
    draw3([[times, solution[:,0]],], [0, 5000], [0.35,0.6], 'x', 'x')
    draw3([[times, solution[:, 1]],], [0, 5000], [0.25,0.3], 'y', 'y')
    f0 = f[0].subs(subs_dict).subs(params[0], params_value[0]).subs(params[1], params_value[1])
    f1 = f[1].subs(subs_dict).subs(params[0], params_value[0]).subs(params[1], params_value[1])
    f0_y = solve(f0, variables[1])[0]
    f0_y = lambdify(variables[0], f0_y, 'numpy')
    f0_x = solve(f0, variables[0])[0]
    f0_x = lambdify(variables[1], f0_x, 'numpy')
    f1_x = solve(f1, variables[0])[0]
    f1_x = lambdify(variables[1], f1_x, 'numpy')
    y_arange = np.arange(0, 1, 0.001)
    x_arange = np.arange(0, 1, 0.001)
    f0_x_arrange = []
    f1_x_arrange = []
    for i in y_arange:
        f0_x_arrange.append(f0_x(i))
        f1_x_arrange.append(f1_x(i))
    f0_y_arrange = []
    for i in x_arange:
        f0_y_arrange.append(f0_y(i))
    sol_cycle = []
    # for i in [1,]:
        # if i%500000==0:
    sol_cycle.append(odeint(oscillation_internal, [0,0], times))
    sol_cycle.append(odeint(oscillation_internal, [0.5,0.25], times))
    sol_cycle.append(odeint(oscillation_internal, [0.25, 0.5], times))
    sol_cycle.append(odeint(oscillation_internal, solution[-1], times))
        # print(sol_cycle)
            # sol_cycle = odeint(oscillation_internal, solution[-1], times)
    # sol_cycle = []
    # plt.plot(x_arange, eq1_y_arrange, linewidth=1.5, label='f1(x,y)=0')
    # plt.plot(eq1_x_arange, y_arange, linewidth=1.5, label='f1(x,y)=0')
    # plt.plot(x_arange, eq2_y_arrange, linestyle='--', linewidth=1.5, label='f2(x,y)=0')
    draw4([[f0_x_arrange, y_arange],[x_arange, f0_y_arrange],[f1_x_arrange, y_arange]], [sol_cycle[-1],], [0.50, 0.56],[0.2, 0.3])
    draw4([[f0_x_arrange, y_arange], [x_arange, f0_y_arrange], [f1_x_arrange, y_arange]], sol_cycle, [0.1,0.9],
          [-0.25,0.75])

x, y, k_3, k_1, k1, k2, k_2, k3 = symbols('x y k_3 k_1 k1 k2 k_2 k3')

exact_values = {k_3: 0.002, k_1: 0.005, k1: 0.12, k2: 1.05, k3: 0.0032}

dxdt = k1 * (1 - x - y) - k_1 * x - k2 * x * ((1 - x - y) ** 2)
dydt = k3 * ((1 - x - y) ** 2) - k_3 * (y**2)

# exact_values = {k_3: 0.001, k_1: 0.01, k1: 0.12, k2: 2.5, k3: 0.0032}
#
#
# dxdt = k1 * (1 - x - 2 * y) - k_1 * x - k2 * x * ((1 - x - 2 *y)**2)
# dydt = k3 * ((1 - x - 2 * y)**2) - k_3 * y


# exact_values = {k_1: 0.04, k1: 1, k_2: 0.02, k3: 10, k2: 1}
#
# dxdt = k1 * (1 - x - y) - k_1 * x - k3 * x * y
# dydt = k2 * ((1 - x - y) ** 2) - k_2 * (y**2) - k3 * x * y

# exact_values = {k_3: 0.003, k_1: 0.03, k1: 0.12, k2: 2, k3: 0.0032}
#
# dxdt = k1 * (1 - x - y) - k_1 * x - k2 * x * ((1 - x - y) ** 2)
# dydt = k3 * (1 - x - y) - k_3 * y


# one_param_analisys([dxdt, dydt], k2)

k_1_values = [0.001,0.005,0.01,0.015,0.02]
for k_1_v in k_1_values:
    exact_values[k_1]=k_1_v
    one_param_analisys([dxdt, dydt], k2)
exact_values[k_1]=0.005

k_3_values = [0.0005,0.001,0.002,0.003,0.004]
for k_3_v in k_3_values:
    exact_values[k_3]=k_3_v
    one_param_analisys([dxdt, dydt], k2)
exact_values[k_3]=0.002

# one_param_analisys([dxdt, dydt], k_1)
two = two_params_analisys2([dxdt, dydt], [x,y], [k1, k_1])
# two = [0.12791999999999998, 0.039466780143593214, 0.3339389366103127]
# two = [0.12791999999999998, 0.03, 0.3]
# k2 = 2;
# k - 3 = 0, 003;
# k3 = 0, 0032;
# k - 1 = 0, 03;
# k1 = 0, 3
auto([dxdt, dydt], [x,y], [k1, k_1], [two[2], two[1]])

# bifurcation.one_parameter_analysis_v1(dxdt, dydt, exact_values, [x, y], k2)
# bifurcation.one_parameter_analysis_v2(dxdt, dydt, exact_values, [x, y], k2)
# bifurcation.two_parameter_analysis(dxdt, dydt, exact_values, [x, y], [k1, k2])