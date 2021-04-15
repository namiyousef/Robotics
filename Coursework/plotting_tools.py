import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import math

# TODO note that this used to be in the main file, the methods of visualiser can be copy pasted into manipulator
# TODO find good design practices! I'm not convinced that it's correct that
# the self's here refer to values from manipulator?

class visualiser:

    def plot_motion(self, x0, xf, timedelta, n_points, manipulator_params, via = [], parametrizer = 'cubic'):
        """ x are cartesian parameters, they contain angles too!"""
        # TODO add code for via points
        X = [x0, via, xf] if via else [x0, xf]
        for x in X:
            if len(x) != 6:
                raise TypeError(f'Your Cartesian parameters must be {self.x_labels}')

        X_params = [self.define_parameter_dict(x, self.x_labels) for x in X]

        Q_params = [self.solve_for_q(x, self.m_params) for x in X_params]
        Q_labels = list(Q_params[-1].keys())

        t = np.linspace(0, timedelta, n_points)

        Q = [np.asarray(list(Q_param.values())).reshape(-1,1) for Q_param in Q_params]

        if parametrizer == 'cubic':
            qi, qdoti, qddoti = self.cubic_interpolator(*Q, t, timedelta)

        # TODO what does this location refer to?
        axis_loc = sym.Matrix([0,0,0,1])

        axis_locs = []
        all_points = np.zeros(
            [4, len(self.T_forwards)]
        )
        fig, ax = plt.subplots()

        """        fig = plt.figure()
        ax = fig.gca(projection='3d')"""

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
                points = np.asarray(axis_locs[j-1:]).T
                ax.plot(*axis_locs[-1][:-1], '.', c = color)

                all_points[:,j] = np.asarray(P).reshape(-1)
                # plot axes
                print(all_points[:-2, j-1:j+1])
                ax.plot(*all_points[:-2, j-1:j+1], 'black', linewidth = 0.2)



        # TODO fix this, currently not looking very great!
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