import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import math

# TODO note that this used to be in the main file, the methods of visualiser can be copy pasted into manipulator
# TODO find good design practices! I'm not convinced that it's correct that
# the self's here refer to values from manipulator?

class interpolations:
    def cubic_interpolator(self, q0, qf, t, td):
        a0 = q0
        a1 = 0
        a2 = (3 / (td ** 2)) * (qf - q0)
        a3 = -(2 / (td ** 3)) * (qf - q0)
        qi = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3
        # TODO qdoti and qddoti are incorrecti
        qdoti = a1 + 2 * a2 * t + 3 * a3 * t ** 2
        qddoti = 2 * a2 + 6 * a3 * t
        return qi, qdoti, qddoti


    def linear_parabolic(self, ):
        pass
class visualiser(interpolations):

    def plot_motion(self, x0, xf, timedelta, n_points, manipulator_params, via = [], via_calculation = "heuristic",parametrizer = 'cubic'):
        """ x are cartesian parameters, they contain angles too!"""
        # TODO add code for via points
        X = [x0, *via, xf] if via else [x0, xf]
        X = [self.check_in_workspace(x) for x in X]


        # TODO via points not implemented!
        # TODO this is a higher level time function, at the end the intervals need to be split too!
        #t = np.linspace(0, timedelta, len(X))


        for x in X:
            if len(x) != 6:
                raise TypeError(f'Your Cartesian parameters must be {self.x_labels}')

        X_params = [self.define_parameter_dict(x, self.x_labels) for x in X]

        Q_params = [self.solve_for_q(x, self.m_params) for x in X_params]



        Q_labels = list(Q_params[-1].keys())


        Q = [np.asarray(list(Q_param.values())).reshape(-1,1) for Q_param in Q_params]
        """if len(X) > 2:
            # calculates the extra conditions on the via points
            V = np.zeros([len(Q), len(Q[0])])
            print(Q)
            for i, q in enumerate(Q):
                # TODO do this in a smart way?
                if not (i == 0 or i == len(X) -1):
                    print(Q[i-1], q, Q[i+1], (q - Q[i-1]), Q[i+1] - q)
                    V[i] = np.where((q - Q[i-1]) * (Q[i+1] - q) < 1, 0, ((q - Q[i-1]) + (Q[i+1] - q))/(t[i] - t[i-1])).reshape(-1)
                    # TODO need to double check if correct!"""



        t = np.linspace(0, timedelta, n_points)

        if parametrizer == 'cubic':
            qi, qdoti, qddoti = self.cubic_interpolator(*Q, t, timedelta)


        self.plot_joint_params(t, qi, qdoti, qddoti)
        # TODO what does this location refer to?
        axis_loc = sym.Matrix([0,0,0,1])

        axis_locs = []
        all_points = np.zeros(
            [4, len(self.T_forwards)]
        )




        fig, ax = plt.subplots()
        fig2, axes = plt.subplots(nrows = 3)
        vels_ = []
        accs_ = []
        locs_ = []
        for i in range(n_points):
            T = 1

            for j, color in enumerate(['tab:blue', 'tab:orange', 'tab:green', 'tab:red']):
                q_temp = {label : val for label, val in zip(Q_labels, qi[:,i])}








                T *= self.T_forwards[j].subs(q_temp).subs(manipulator_params)
                # TODO need to improve this portion of the code, make more consistent and nice?
                P = T * axis_loc
                axis_locs.append(
                    tuple(p for p in P[:-1])
                )
                ax.plot(*axis_locs[-1][:-1], '.', c = color)

                all_points[:,j] = np.asarray(P).reshape(-1)
                ax.plot(*all_points[:-2, j-1:j+1], 'black', linewidth = 0.2, label = '__nolegend__')

            # remember, you only do this for FINAL point!
            # TODO this is jacobian change it make more adaptable
            j_, jdot_ = self.my_jacobian(qi[:, i], qdoti[:, i])
            vels = np.dot(j_, qdoti[:, i])
            vels_.append(vels)
            plot = 0
            if plot:
                ax.plot(*axis_locs[-1][:-1], '.', c='tab:red')
                try:
                    ax.quiver(float(axis_locs[-1][:-1][0]),
                               float(axis_locs[-1][:-1][1]),
                               float(vels[0]),
                               float(vels[1]), scale = 5,  linewidths = 1)


                    print(vels[0],'\n', vels[2])
                    print('=============')
                    print(
                               float(vels[0]),'\n',
                               float(vels[2]))

                except:

                    print(float(vels[0]))
                    return

            accs = np.dot(jdot_, qdoti[:, i]) + np.dot(j_, qddoti[:, i])
            accs_.append(accs)
            locs_.append(axis_locs[-1][:-1])

        print(locs_)
        print(t)
        modulus = lambda a : a[0]**2 + a[1]**2
        axes[0].plot(t, [modulus(loc_) for loc_ in locs_], 'o-', c = 'tab:red')
        axes[1].plot(t, [modulus(loc_) for loc_ in vels_], 'o-', c = 'tab:red')
        axes[2].plot(t, [modulus(loc_) for loc_ in accs_], 'o-', c = 'tab:red')
        # TODO fix this, currently not looking very great!
        plt.show()

    def plot_joint_params(self, t, qi, qdoti, qddoti, colors = ['tab:blue', 'tab:orange', 'tab:green'], labels = ['d2', 'theta1', 'theta3']):
        fig, axes = plt.subplots(nrows = 3)
        Q = [qi, qdoti, qddoti]
        plt.suptitle('Joint parameter plots against time')
        for q,ax,label in zip(Q, axes, ['values', 'velocities', 'accelerations']):
            for q_ in q:
                ax.plot(t, q_)
                ax.set_ylabel(f'parameter {label}')

        plt.legend(labels)

        plt.show()
            # TODO find the joint parameters
            # parametrize them using the relevant function
            # save that
        # needs to give orientation, how do you generalise this?
        pass
        # very smart! How can I take advantage of the subs??

    def plot_workspace(self, mode = 'points', n_points = 100):
        # TODO make sure this is generlaised,!
        Q_labels = [joint.q_param for joint in self.joints[:-1]]
        Q = [np.linspace(int(joint.constraint[0]), int(joint.constraint[1]), n_points) for joint in self.joints[:-1]]


        if mode == 'points':
            Q = np.meshgrid(*Q, indexing='ij')
            Q = [q.reshape(-1) for q in Q]
            # low perf
            #Q = np.asarray(Q).T
            #X = np.zeros(Q.shape)
            #for i, q in enumerate(Q):
            #    q_temp = {label: val for label, val in zip(Q_labels, q)}
            #    x = self.T_FK[:-1, -1]
            #    X[i] = np.asarray(x.subs(self.m_params).subs(q_temp)).reshape(-1)
            #ax.scatter3D(*X.T, alpha = 0.2)

            pxa = np.cos(Q[0] + Q[2]) + math.cos(math.pi / 4) * np.sin(Q[0]) * (1 + Q[1])
            pya = np.sin(Q[0] + Q[2]) - math.cos(math.pi / 4) * np.cos(Q[0]) * (1 + Q[1])
            pza = 1 + (1 + Q[1]) * math.sin(math.pi / 4)
            fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
            ax.scatter3D(pxa, pya, pza, alpha=0.2)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            # TODO need to improve this portion of the code, make more consistent and nice?
            plt.show()
        else:
            # first two are indices to use, last one is index to turn off
            # NOT WORKING
            Q = np.meshgrid(0, indexing='ij')
            pxa = np.cos(Q[0] + Q[2]) + math.cos(math.pi/4) * np.sin(Q[0]) * (1 + Q[1])
            pya = np.sin(Q[0] + Q[2]) - math.cos(math.pi/4) * np.cos(Q[0]) * (1 + Q[1])
            pza = 1 + (1 + Q[1]) * math.sin(math.pi/4)




            trace1 = go.Surface(z=pza, x=pxa, y=pya,
                                colorscale='Reds',
                                showscale=False,
                                opacity=0.7,
                                )
            data = [trace1]


            layout = go.Layout(scene=dict(
                xaxis=dict(title='x (m)'),
                yaxis=dict(title='y (m)'),
                zaxis=dict(title='z (m)'),
            ),
            )


            fig = go.Figure(data=data, layout=layout)
            fig.show()
        # TODO incomplete
    def generate_workspace(self, constraints):
        # TODO generate a meshgrid from constraints
        # TODO use inverse kinematics to generate the workspace
        pass

    # can you make plotting inheritance smarter? maybe interpolations or visualiser can store
    # the values that you are interested in? instead of having to incorporate the plots inside the
    # main plot motion function?
    def plot_velocity(self, j):
        pass

    def plot_acceleration(self):
        pass

    def my_jacobian(self, q, qdot):
        # these are fixed!
        d2, theta1, theta3 = q
        d2dot, theta1dot, theta2dot = qdot

        c_phi = math.cos(math.pi/4)
        s_phi = math.sin(math.pi/4)
        gamma = theta1 + theta3
        y = math.sin(gamma) - c_phi*math.sin(theta1)*(1+d2)
        x = math.cos(gamma) + c_phi * math.sin(theta1) * (1 + d2)

        j = np.array([
            [-y, c_phi*math.sin(theta1), -math.sin(gamma)],
            [x, -c_phi*math.cos(theta1), math.cos(gamma)],
            [0,0,s_phi],
            [0,0,0],
            [0,0,0],
            [1,0,1]
        ])
        gammadot = theta2dot+ theta1dot
        xdot = np.dot(j[0], qdot)
        ydot = np.dot(j[1], qdot)

        j_2 = np.array([
            [-ydot, theta1dot * c_phi * math.cos(theta1), -math.cos(gamma)*(gammadot)],
            [xdot, theta1dot * c_phi * math.sin(theta1), -math.sin(gamma)*gammadot],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ])
        return j, j_2