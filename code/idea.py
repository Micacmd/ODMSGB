import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from sklearn.cluster import k_means
from sklearn.metrics import roc_curve, auc


def get_radius(gb):
    real_gb = gb[:, :-1]
    center = real_gb.mean(0)
    radius = max(np.sqrt(np.sum((real_gb - center) ** 2, axis=1)))
    return radius


def spilt_ball(data):
    m = data.shape[1]
    cluster = k_means(X=data[:, :m - 1], init='k-means++', n_clusters=2, n_init=5)[1]
    ball1 = data[cluster == 0, :]
    ball2 = data[cluster == 1, :]
    return [ball1, ball2]


def spilt_ball_2(data):
    real_data = data[:, :-1]
    ball1 = []
    ball2 = []
    D = cdist(real_data, real_data)
    r, c = np.where(D == np.max(D))
    r1 = r[1]
    c1 = c[1]
    for j in range(0, len(data)):
        if D[j, r1] < D[j, c1]:
            ball1.extend([data[j, :]])
        else:
            ball2.extend([data[j, :]])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]


def get_density_volume(gb):
    real_gb = gb[:, :-1]
    num = len(real_gb)
    center = real_gb.mean(0)
    sum_radius = np.sum(np.sqrt(np.sum((real_gb - center) ** 2, axis=1)))
    mean_radius = sum_radius / num
    if mean_radius != 0:
        density_volume = num / sum_radius
    else:
        density_volume = num

    return density_volume


def division_ball(gb_list):
    gb_list_new = []
    for gb in gb_list:
        if len(gb) >= 4:
            ball_1, ball_2 = spilt_ball_2(gb)
            if len(ball_1) == 0 or len(ball_2) == 0:
                gb_list_new.append(gb)
                continue
            density_parent = get_density_volume(gb)
            density_child_1 = get_density_volume(ball_1)
            density_child_2 = get_density_volume(ball_2)
            w = len(ball_1) + len(ball_2)
            w1 = len(ball_1) / w
            w2 = len(ball_2) / w
            w_child = (w1 * density_child_1 + w2 * density_child_2)
            t2 = (w_child > density_parent)
            if t2:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_new.append(gb)
        else:
            gb_list_new.append(gb)

    return gb_list_new


def normalized_ball(gb_list, radius_detect):
    gb_list_temp = []
    for gb in gb_list:
        if len(gb) < 2:
            gb_list_temp.append(gb)
        else:
            ball_1, ball_2 = spilt_ball_2(gb)
            if get_radius(gb) <= 2 * radius_detect:
                gb_list_temp.append(gb)
            else:
                gb_list_temp.extend([ball_1, ball_2])

    return gb_list_temp


def get_GB(data):
    data_num = data.shape[0]
    index = np.array(range(data_num)).reshape(data_num, 1)
    data = np.hstack((data, index))
    gb_list_temp = [data]
    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = division_ball(gb_list_temp)
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break
    radius = []
    for gb in gb_list_temp:
        if len(gb) >= 2:
            radius.append(get_radius(gb))

    radius_median = np.median(radius)
    radius_mean = np.mean(radius)
    radius_detect = max(radius_median, radius_mean)

    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = normalized_ball(gb_list_temp, radius_detect)
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break

    gb_list_final = gb_list_temp
    return gb_list_final
def matrix_operation(A, B):
    return np.minimum(A, B)


def matrix_operations(matrices):
    n = matrices.shape[2]  # 获取矩阵的数量
    m, n_cols = matrices.shape[:2]  # 获取矩阵的大小
    results = np.zeros((m, n_cols, n * n))  # 初始化结果矩阵

    index = 0
    for i in range(n):
        for j in range(n):
            result = matrix_operation(matrices[:, :, i], matrices[:, :, j])
            results[:, :, index] = result
            index += 1

    return results

def idea(data, sigma):
    n, m = data.shape
    GBs = get_GB(data)
    n_gb = len(GBs)
    centers = np.zeros((n_gb, m))
    for idx, gb in enumerate(GBs):
        centers[idx] = np.mean(gb[:, :-1], axis=0)

    En = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            r1 = 1 / (1 + cdist(centers[:, [i]], centers[:, [i]]))
            r1[r1 < sigma] = 0
            r2 = 1 / (1 + cdist(centers[:, [j]], centers[:, [j]]))
            r2[r2 < sigma] = 0
            # 使用matrix_operation计算联合熵
            R_sum = matrix_operation(r1, r2)  # 计算联合矩阵
            En[i,j] = -(1 / n_gb) * np.sum(np.log2(np.sum(R_sum, axis=1) / n_gb))

    RE = np.zeros((n, m))
    for i in range(m):
        En_de = np.argsort(-En[i, :])  # 降序
        En_as = np.argsort(En[i, :])  # 升序

        weight = np.zeros((n_gb, m))
        EN_de = np.zeros((n_gb, m))
        EN_as = np.zeros((n_gb, m))

        for k in range(m):
            FSet = 1 / (1 + cdist(centers[:, [k]], centers[:, [k]]))
            FSet_de = 1 / (1 + cdist(centers[:, En_de[:m - k]], centers[:, En_de[:m - k]]))
            FSet_as = 1 / (1 + cdist(centers[:, En_as[:m - k]], centers[:, En_as[:m - k]]))

            FSet[FSet < sigma] = 0
            FSet_de[FSet_de < sigma] = 0
            FSet_as[FSet_as < sigma] = 0

            for l in range(n_gb):
                weight[l, k] = np.sum(FSet[l, :]) / n_gb
                EN_de[l, k] = np.sum(FSet_de[l, :]) / n_gb
                EN_as[l, k] = np.sum(FSet_as[l, :]) / n_gb

        OF_gb = np.zeros(n_gb)
        for j in range(n_gb):
            OF_gb = 1 - (np.cbrt(np.sum(weight) / m)) * ((np.sum(EN_as + EN_de,axis=1)) / (2 * m))
        OF = np.zeros(n)
        for idx, gb in enumerate(GBs):
            point_idxs = gb[:, -1].astype('int')
            OF[point_idxs] = OF_gb[idx]
        RE[:, i] = OF

    return np.sum(RE, axis=1) / m


if __name__ == '__main__':
    load_data = loadmat('yeast_ERL_5_variant1.mat')
    trandata = load_data['trandata']
    labels = trandata[:, -1]

    idea_out_scores_T = idea(trandata[:, :-1], 0.5)

    idea_out_scores = idea_out_scores_T.T

    labels_full = trandata[:, -1]

    # 计算ROC和AUC
    FPR_full, TPR_full, _ = roc_curve(labels_full, idea_out_scores_T, pos_label=1)
    AUC_full = auc(FPR_full, TPR_full)

    print('AUC of full data:', AUC_full)
