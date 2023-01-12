import numpy as np
import numpy as np
from cvxopt import matrix, solvers
from numpy import genfromtxt


def compute_weight_with_fairness_with_label(loss_vector, lower_bound, upper_bound, Theta_x, Sen, cluster):
    ### quardtic coefficient

    Q = 2 * np.zeros((len(cluster), len(cluster)))
    Q = Q.astype('double')

    q_matrix = np.zeros(len(cluster))
    index = 0
    for i in range(len(q_matrix)):
        count = cluster[i]
        for j in range(index, index + count):
            q_matrix[i] += loss_vector[j] /len(loss_vector)
        index = index + count
    q_matrix = q_matrix.reshape(-1, 1)
    q = q_matrix.astype('double')




    ### Compute A, b equality constraint
    A = np.zeros((1, len(cluster)))
    b = [0.0 * len(cluster)]
    A = A.astype('double')

    ### compute fariness constraint
    ### compute first term of fairness constraint
    tau = 0.1
    ### compute fariness constraint
    ### compute first term of fairness constraint

    sen_bar = np.sum(Sen) / len(Sen)

    Theta_x = np.sum(Theta_x, axis=1)
    Theta_x = Theta_x.reshape(len(Theta_x), 1)
    Theta_x = Theta_x/max(Theta_x)

    fair = np.multiply(Sen - sen_bar, Theta_x)
    fair_cons = np.zeros(len(cluster))
    index = 0
    for i in range(len(fair_cons)):
        count = cluster[i]
        for j in range(index, index + count):
            fair_cons[i] += (fair[j]/len(loss_vector))
        index = index + count

    # print("theta")
    # print(Theta_x)

    # print("fair")
    # print(fair_cons)
    fair = fair_cons.reshape(-1, 1)


    # ### G(w, alpha) < tau and G(w, alpha) > -tau
    fair_G = np.concatenate((fair.transpose(), -fair.transpose()))
    fair_h = np.full((2, 1), tau)


    # ### compute G and h inequality constraint and combine with fairness inequality
    lower = -np.identity(len(cluster))
    upper = np.identity(len(cluster))
    G = np.concatenate((lower, upper), axis=0)


    G = np.concatenate((G, fair_G))
    #

    h = np.concatenate((lower_bound, upper_bound), axis=0)
    h = h.reshape(len(h), 1)

    # print("lower bound")
    # print(lower_bound)
    # print(upper_bound)


    h = np.concatenate((h, fair_h), axis=0)
    h = h.astype('double')

    Q, p, G, h, A, b = -matrix(Q), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b)
    #
    sol = solvers.qp(Q, p, G, h)
    sol = np.array(sol['x'])

    # print("solution")
    # print(sol)
    weight = np.zeros(len(Sen))
    index = 0
    max_upper = max(upper_bound)
    min_lower = min(lower_bound)
    for i in range(len(cluster)):
        count = cluster[i]
        for j in range(index, index + count):
            weight[j] = abs(sol[i])
            if weight[j] > max_upper:
                weight[j] = max_upper
            if weight[j] < min_lower:
                weight[j] = min_lower
        index = index + count

    weight = weight.reshape(len(weight), 1)


    ratio = np.sum(weight * Sen) / np.sum(weight)

    return weight, ratio







def compute_weight_with_fairness_no_label(loss_vector, lower_bound, upper_bound, Theta_x, Sen, cluster):
    solvers.options['show_progress'] = False
    Q = 2 * np.zeros((len(cluster), len(cluster)))
    Q = Q.astype('double')

    q_matrix = np.zeros(len(cluster))
    index = 0
    for i in range(len(q_matrix)):
        count = cluster[i]
        for j in range(index, index + count):
            q_matrix[i] += loss_vector[j] / len(loss_vector)
        index = index + count
    q_matrix = q_matrix.reshape(-1, 1)
    q = q_matrix.astype('double')

    ### Compute A, b equality constraint
    A = np.zeros((1, len(cluster)))
    b = [0.0 * len(cluster)]
    A = A.astype('double')

    ### compute fariness constraint
    ### compute first term of fairness constraint
    tau = 0.1
    ### compute fariness constraint
    ### compute first term of fairness constraint

    sen_bar = np.sum(Sen) / len(Sen)

    Theta_x = np.sum(Theta_x, axis=1)
    Theta_x = Theta_x.reshape(len(Theta_x), 1)
    Theta_x = Theta_x / max(Theta_x)

    fair = np.multiply(Sen - sen_bar, Theta_x)
    fair_cons = np.zeros(len(cluster))
    index = 0
    for i in range(len(fair_cons)):
        count = cluster[i]
        for j in range(index, index + count):
            fair_cons[i] += (fair[j] / len(loss_vector))
        index = index + count

    # print("theta")
    # print(Theta_x)

    # print("fair")
    # print(fair_cons)
    fair = fair_cons.reshape(-1, 1)

    # ### G(w, alpha) < tau and G(w, alpha) > -tau
    fair_G = np.concatenate((fair.transpose(), -fair.transpose()))
    fair_h = np.full((2, 1), tau)

    # ### compute G and h inequality constraint and combine with fairness inequality
    lower = -np.identity(len(cluster))
    upper = np.identity(len(cluster))
    G = np.concatenate((lower, upper), axis=0)

    G = np.concatenate((G, fair_G))
    #

    h = np.concatenate((lower_bound, upper_bound), axis=0)
    h = h.reshape(len(h), 1)

    # print("lower bound")
    # print(lower_bound)
    # print(upper_bound)

    h = np.concatenate((h, fair_h), axis=0)
    h = h.astype('double')

    Q, p, G, h, A, b = -matrix(Q), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b)
    #
    sol = solvers.qp(Q, p, G, h)
    sol = np.array(sol['x'])

    # print("solution")
    # print(sol)
    weight = np.zeros(len(Sen))
    index = 0
    max_upper = max(upper_bound)
    min_lower = min(lower_bound)
    for i in range(len(cluster)):
        count = cluster[i]
        for j in range(index, index + count):
            weight[j] = abs(sol[i])
            if weight[j] > max_upper:
                weight[j] = max_upper
            if weight[j] < min_lower:
                weight[j] = min_lower
        index = index + count

    weight = weight.reshape(len(weight), 1)

    ratio = np.sum(weight * Sen) / np.sum(weight)

    return weight, ratio