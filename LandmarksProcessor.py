import math
import cv2
import numpy as np
from enum import IntEnum


class FaceType(IntEnum):
    # enumerating in order "next contains prev"
    HALF = 0
    MID_FULL = 1
    FULL = 2
    FULL_NO_ALIGN = 3
    WHOLE_FACE = 4
    HEAD = 10
    HEAD_NO_ALIGN = 20

    MARK_ONLY = 100,  # no align at all, just embedded faceinfo

    @staticmethod
    def fromString(s):
        r = from_string_dict.get(s.lower())
        if r is None:
            raise Exception('FaceType.fromString value error')
        return r

    @staticmethod
    def toString(face_type):
        return to_string_dict[face_type]


to_string_dict = {FaceType.HALF: 'half_face',
                  FaceType.MID_FULL: 'midfull_face',
                  FaceType.FULL: 'full_face',
                  FaceType.FULL_NO_ALIGN: 'full_face_no_align',
                  FaceType.WHOLE_FACE: 'whole_face',
                  FaceType.HEAD: 'head',
                  FaceType.HEAD_NO_ALIGN: 'head_no_align',

                  FaceType.MARK_ONLY: 'mark_only',
                  }

from_string_dict = {to_string_dict[x]: x for x in to_string_dict.keys()}


def polygon_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def umeyama(src, dst, estimate_scale):
    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T


landmarks_2D_new = np.array([
    [0.000213256, 0.106454],  # 17
    [0.0752622, 0.038915],  # 18
    [0.18113, 0.0187482],  # 19
    [0.29077, 0.0344891],  # 20
    [0.393397, 0.0773906],  # 21
    [0.586856, 0.0773906],  # 22
    [0.689483, 0.0344891],  # 23
    [0.799124, 0.0187482],  # 24
    [0.904991, 0.038915],  # 25
    [0.98004, 0.106454],  # 26
    [0.490127, 0.203352],  # 27
    [0.490127, 0.307009],  # 28
    [0.490127, 0.409805],  # 29
    [0.490127, 0.515625],  # 30
    [0.36688, 0.587326],  # 31
    [0.426036, 0.609345],  # 32
    [0.490127, 0.628106],  # 33
    [0.554217, 0.609345],  # 34
    [0.613373, 0.587326],  # 35
    [0.121737, 0.216423],  # 36
    [0.187122, 0.178758],  # 37
    [0.265825, 0.179852],  # 38
    [0.334606, 0.231733],  # 39
    [0.260918, 0.245099],  # 40
    [0.182743, 0.244077],  # 41
    [0.645647, 0.231733],  # 42
    [0.714428, 0.179852],  # 43
    [0.793132, 0.178758],  # 44
    [0.858516, 0.216423],  # 45
    [0.79751, 0.244077],  # 46
    [0.719335, 0.245099],  # 47
    [0.254149, 0.780233],  # 48
    [0.726104, 0.780233],  # 54
], dtype=np.float32)

# 68 point landmark definitions
landmarks_68_pt = {"mouth": (48, 68),
                   "right_eyebrow": (17, 22),
                   "left_eyebrow": (22, 27),
                   "right_eye": (36, 42),
                   "left_eye": (42, 48),
                   "nose": (27, 36),  # missed one point
                   "jaw": (0, 17)}

FaceType_to_padding_remove_align = {
    FaceType.HALF: (0.0, False),
    FaceType.MID_FULL: (0.0675, False),
    FaceType.FULL: (0.2109375, False),
    FaceType.FULL_NO_ALIGN: (0.2109375, True),
    FaceType.WHOLE_FACE: (0.35, False),
    FaceType.HEAD: (0.70, False),
    FaceType.HEAD_NO_ALIGN: (0.70, True),
}


# wf默认0.4
def convert_98_to_68(lmrks):
    # jaw
    result = [lmrks[0]]
    for i in range(2, 16, 2):
        result += [(lmrks[i] + (lmrks[i - 1] + lmrks[i + 1]) / 2) / 2]
    result += [lmrks[16]]
    for i in range(18, 32, 2):
        result += [(lmrks[i] + (lmrks[i - 1] + lmrks[i + 1]) / 2) / 2]
    result += [lmrks[32]]

    # eyebrows averaging
    result += [lmrks[33],
               (lmrks[34] + lmrks[41]) / 2,
               (lmrks[35] + lmrks[40]) / 2,
               (lmrks[36] + lmrks[39]) / 2,
               (lmrks[37] + lmrks[38]) / 2,
               ]

    result += [(lmrks[42] + lmrks[50]) / 2,
               (lmrks[43] + lmrks[49]) / 2,
               (lmrks[44] + lmrks[48]) / 2,
               (lmrks[45] + lmrks[47]) / 2,
               lmrks[46]
               ]

    # nose
    result += list(lmrks[51:60])

    # left eye (from our view)
    result += [lmrks[60],
               lmrks[61],
               lmrks[63],
               lmrks[64],
               lmrks[65],
               lmrks[67]]

    # right eye
    result += [lmrks[68],
               lmrks[69],
               lmrks[71],
               lmrks[72],
               lmrks[73],
               lmrks[75]]

    # mouth
    result += list(lmrks[76:96])

    return np.concatenate(result).reshape((68, 2))


def transform_points(points, mat, invert=False):
    if invert:
        mat = cv2.invertAffineTransform(mat)
    ''' transform_points 函数接受输入的坐标点 points、仿射变换矩阵 mat 和一个布尔值 invert，用于确定是否反转（逆转）仿射变换。'''
    points = np.expand_dims(points, axis=1)
    '''将输入的坐标点 points 在第一个维度上添加一个维度，从而将其形状从 (n, 2) 转换为 (n, 1, 2)。'''
    points = cv2.transform(points, mat, points.shape)
    '''使用 OpenCV 的 cv2.transform 函数将坐标点 points 应用到仿射变换矩阵 mat 上，得到变换后的坐标点'''
    points = np.squeeze(points)
    '''将变换后的坐标点 points 的形状从 (n, 1, 2) 转换为 (n, 2)，去除多余的维度。'''
    return points


def get_transform_mat(image_landmarks, output_size, face_type, scale=1.0):
    """image_landmarks: 图像关键点、output_size: 输出大小、face_type: 脸部类型、scale: 缩放比例、return: 变换矩阵"""

    # 如果输入不是NumPy数组，则将其转换为NumPy数组。
    if not isinstance(image_landmarks, np.ndarray):
        image_landmarks = np.array(image_landmarks)

    # 使用Umeyama算法计算从全局空间到局部对齐空间的变换矩阵。
    mat = umeyama(np.concatenate([image_landmarks[17:49], image_landmarks[54:55]]), landmarks_2D_new, True)[0:2]

    '''g_p 是通过将坐标点 [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)] 使用变换矩阵 mat 进行变换得到的全局坐标点。g_c 是 g_p 中索引为 4 的全局坐标点。'''
    g_p = transform_points(np.float32([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]), mat, True)
    g_c = g_p[4]

    # calc diagonal vectors between corners in global space
    # 计算对角向量并规范化。
    tb_diag_vec = (g_p[2] - g_p[0]).astype(np.float32)
    tb_diag_vec /= np.linalg.norm(tb_diag_vec)
    '''tb_diag_vec 是全局坐标点 g_p 中索引 2 和 0 对应的向量，并进行了归一化处理。'''
    bt_diag_vec = (g_p[1] - g_p[3]).astype(np.float32)
    bt_diag_vec /= np.linalg.norm(bt_diag_vec)
    '''bt_diag_vec 是全局坐标点 g_p 中索引 1 和 3 对应的向量，并进行了归一化处理。'''

    # calc modifier of diagonal vectors for scale and padding value
    # 计算缩放因子和填充值。
    padding, remove_align = FaceType_to_padding_remove_align.get(face_type, 0.0)
    '''padding 和 remove_align 是根据 face_type 从 FaceType_to_padding_remove_align 字典中获取的填充和移除对齐系数。'''
    mod = (1.0 / scale) * (np.linalg.norm(g_p[0] - g_p[2]) * (padding * np.sqrt(2.0) + 0.5))
    ''' mod 是根据缩放因子 scale 和全局坐标点 g_p 中索引 0 和 2 对应向量的长度计算得到的值。'''

    if face_type == FaceType.WHOLE_FACE:
        # 根据脸型类型调整垂直和水平偏移量。
        vec = (g_p[0] - g_p[3]).astype(np.float32)
        vec_len = np.linalg.norm(vec)
        vec /= vec_len
        g_c += vec * vec_len * 0.07  # 将 g_c 偏移 vec * vec_len * 0.07 的距离，以覆盖额头下方更多区域。

    elif face_type == FaceType.HEAD:
        # 假设 image_landmarks 变量已经包含了头部的3D特征点信息，根据估计的头部左右转角度数（yaw），调整水平偏移量。
        yaw = estimate_averaged_yaw(transform_points(image_landmarks, mat, False))

        hvec = (g_p[0] - g_p[1]).astype(np.float32)
        hvec_len = np.linalg.norm(hvec)
        hvec /= hvec_len

        yaw *= np.abs(math.tanh(yaw * 2))  # Damp near zero

        g_c -= hvec * (yaw * hvec_len / 2.0)

        # adjust vertical offset for HEAD, 50% below
        vvec = (g_p[0] - g_p[3]).astype(np.float32)
        vvec_len = np.linalg.norm(vvec)
        vvec /= vvec_len
        g_c += vvec * vvec_len * 0.50

    # calc 3 points in global space to estimate 2d affine transform
    if not remove_align:
        l_t = np.array([g_c - tb_diag_vec * mod,
                        g_c + bt_diag_vec * mod,
                        g_c + tb_diag_vec * mod])
    else:
        # remove_align - face will be centered in the frame but not aligned
        l_t = np.array([g_c - tb_diag_vec * mod,
                        g_c + bt_diag_vec * mod,
                        g_c + tb_diag_vec * mod,
                        g_c - bt_diag_vec * mod,
                        ])

        # get area of face square in global space
        area = polygon_area(l_t[:, 0], l_t[:, 1])

        # calc side of square
        side = np.float32(math.sqrt(area) / 2)

        # calc 3 points with unrotated square
        l_t = np.array([g_c + [-side, -side],
                        g_c + [side, -side],
                        g_c + [side, side]])

    # calc affine transform from 3 global space points to 3 local space points size of 'output_size'
    # 计算仿射变换矩阵，将全局空间中的三个点映射到本地对齐空间中的三个点。返回变换矩阵
    pts2 = np.float32(((0, 0), (output_size, 0), (output_size, output_size)))
    mat = cv2.getAffineTransform(l_t, pts2)
    return mat


def expand_eyebrows(lmrks, eyebrows_expand_mod=1.0):
    if len(lmrks) != 68:
        raise Exception('works only with 68 landmarks')
    lmrks = np.array(lmrks.copy(), dtype=np.int32)

    # #nose
    ml_pnt = (lmrks[36] + lmrks[0]) // 2
    mr_pnt = (lmrks[16] + lmrks[45]) // 2

    # mid points between the mid points and eye
    ql_pnt = (lmrks[36] + ml_pnt) // 2
    qr_pnt = (lmrks[45] + mr_pnt) // 2

    # Top of the eye arrays
    bot_l = np.array((ql_pnt, lmrks[36], lmrks[37], lmrks[38], lmrks[39]))
    bot_r = np.array((lmrks[42], lmrks[43], lmrks[44], lmrks[45], qr_pnt))

    # Eyebrow arrays
    top_l = lmrks[17:22]
    top_r = lmrks[22:27]

    # Adjust eyebrow arrays
    lmrks[17:22] = top_l + eyebrows_expand_mod * 0.5 * (top_l - bot_l)
    lmrks[22:27] = top_r + eyebrows_expand_mod * 0.5 * (top_r - bot_r)
    return lmrks


def estimate_averaged_yaw(landmarks):
    # Works much better than solvePnP if landmarks from "3DFAN"
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
    l = ((landmarks[27][0] - landmarks[0][0]) + (landmarks[28][0] - landmarks[1][0]) + (
            landmarks[29][0] - landmarks[2][0])) / 3.0
    r = ((landmarks[16][0] - landmarks[27][0]) + (landmarks[15][0] - landmarks[28][0]) + (
            landmarks[14][0] - landmarks[29][0])) / 3.0
    return float(r - l)


def draw_landmarks(image, image_landmarks, color=(0, 255, 0), draw_circles=True, thickness=1, transparent_mask=False):
    if len(image_landmarks) != 68:
        raise Exception('get_image_eye_mask works only with 68 landmarks')

    int_lmrks = np.array(image_landmarks, dtype=np.int32)

    jaw = int_lmrks[slice(*landmarks_68_pt["jaw"])]
    right_eyebrow = int_lmrks[slice(*landmarks_68_pt["right_eyebrow"])]
    left_eyebrow = int_lmrks[slice(*landmarks_68_pt["left_eyebrow"])]
    mouth = int_lmrks[slice(*landmarks_68_pt["mouth"])]
    right_eye = int_lmrks[slice(*landmarks_68_pt["right_eye"])]
    left_eye = int_lmrks[slice(*landmarks_68_pt["left_eye"])]
    nose = int_lmrks[slice(*landmarks_68_pt["nose"])]

    # open shapes
    cv2.polylines(image,
                  tuple(np.array([v]) for v in (right_eyebrow, jaw, left_eyebrow, np.concatenate((nose, [nose[-6]])))),
                  False, color, thickness=thickness, lineType=cv2.LINE_AA)
    # closed shapes
    cv2.polylines(image, tuple(np.array([v]) for v in (right_eye, left_eye, mouth)),
                  True, color, thickness=thickness, lineType=cv2.LINE_AA)

    if draw_circles:
        # the rest of the cicles
        for x, y in np.concatenate((right_eyebrow, left_eyebrow, mouth, right_eye, left_eye, nose), axis=0):
            cv2.circle(image, (x, y), 1, color, 1, lineType=cv2.LINE_AA)
        # jaw big circles
        for x, y in jaw:
            cv2.circle(image, (x, y), 2, color, lineType=cv2.LINE_AA)

    if transparent_mask:
        mask = get_image_hull_mask(image.shape, image_landmarks)
        image[...] = (image * (1 - mask) + image * mask / 2)[...]


def get_image_hull_mask(image_shape, image_landmarks, eyebrows_expand_mod=1.0):
    hull_mask = np.zeros(image_shape[0:2] + (1,), dtype=np.float32)

    lmrks = expand_eyebrows(image_landmarks, eyebrows_expand_mod)

    r_jaw = (lmrks[0:9], lmrks[17:18])
    l_jaw = (lmrks[8:17], lmrks[26:27])
    r_cheek = (lmrks[17:20], lmrks[8:9])
    l_cheek = (lmrks[24:27], lmrks[8:9])
    nose_ridge = (lmrks[19:25], lmrks[8:9],)
    r_eye = (lmrks[17:22], lmrks[27:28], lmrks[31:36], lmrks[8:9])
    l_eye = (lmrks[22:27], lmrks[27:28], lmrks[31:36], lmrks[8:9])
    nose = (lmrks[27:31], lmrks[31:36])
    parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]

    for item in parts:
        merged = np.concatenate(item)
        cv2.fillConvexPoly(hull_mask, cv2.convexHull(merged), (1,))

    return hull_mask


def draw_rect_landmarks(image, rect, image_landmarks, face_type, face_size=256, transparent_mask=False,
                        landmarks_color=(0, 255, 0)):
    draw_landmarks(image, image_landmarks, color=landmarks_color, transparent_mask=transparent_mask)
    draw_rect(image, rect, (255, 0, 0), 2)

    image_to_face_mat = get_transform_mat(image_landmarks, face_size, face_type)
    points = transform_points([(0, 0), (0, face_size - 1), (face_size - 1, face_size - 1), (face_size - 1, 0)],
                              image_to_face_mat, True)
    draw_polygon(image, points, (0, 0, 255), 2)

    points = transform_points(
        [(int(face_size * 0.05), 0), (int(face_size * 0.1), int(face_size * 0.1)), (0, int(face_size * 0.1))],
        image_to_face_mat, True)
    draw_polygon(image, points, (0, 0, 255), 2)


def draw_polygon(image, points, color, thickness=1):
    points_len = len(points)
    for i in range(0, points_len):
        p0 = tuple(points[i])
        p1 = tuple(points[(i + 1) % points_len])
        cv2.line(image, p0, p1, color, thickness=thickness)


def draw_rect(image, rect, color, thickness=1):
    l, t, r, b = rect
    draw_polygon(image, [(l, t), (r, t), (r, b), (l, b)], color, thickness)
