# -*- coding: utf-8 -*-
from pathlib import Path
import os
import tqdm
import time
from DFLJPG import *
from LandmarksProcessor import *
from FaceType import *
from insightface.app import FaceAnalysis


class extract:
    def __init__(self, original_folder, aligned_folder, face_style, jpeg_quality,
                 face_num, need_debug, max_worker, image_size, det_size):
        self.num = 1
        self.original_folder = original_folder  # 原图文件夹
        self.aligned_folder = aligned_folder  # 保存人脸文件夹
        self.face_style = {'wf': FaceType.WHOLE_FACE,
                           'head': FaceType.HEAD,
                           'f': FaceType.FULL}[face_style]
        self.jpeg_quality = int(jpeg_quality)
        self.face_num = int(face_num)
        self.need_debug = need_debug
        self.max_worker = int(max_worker)
        self.image_size = int(image_size)
        self.det_size = int(det_size)
        self.text = []
        self.faceAnalysis = FaceAnalysis(
            allowed_modules=['detection', 'landmark_2d_106'])  # landmark_2d_106
        self.faceAnalysis.prepare(
            ctx_id=0, det_thresh=0.6, det_size=(self.det_size, self.det_size))  # ctx_id=0表示选显卡0

        os.makedirs(self.aligned_folder, exist_ok=True)

    # pyqt5用cv2很难保存，需要改写
    @staticmethod
    def cv2_imwrite(filename, img, *args):
        ret, buf = cv2.imencode(Path(filename).suffix, img, *args)
        if ret:
            try:
                with open(filename, "wb") as stream:
                    stream.write(buf)
            except:
                pass

    # 转68特征点
    @staticmethod
    def landmark106to68(pt106):
        landmark106to68 = [1, 10, 12, 14, 16, 3, 5, 7, 0, 23, 21, 19, 32, 30, 28, 26, 17,  # 脸颊17点
                           43, 48, 49, 51, 50,  # 左眉毛5点
                           102, 103, 104, 105, 101,  # 右眉毛5点
                           72, 73, 74, 86, 78, 79, 80, 85, 84,  # 鼻子9点
                           35, 41, 42, 39, 37, 36,  # 左眼睛6点
                           89, 95, 96, 93, 91, 90,  # 右眼睛6点
                           52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55, 65, 66, 62, 70, 69, 57, 60, 54  # 嘴巴20点
                           ]
        pt68 = []
        if len(pt106) != 106:
            return None
        for i in range(68):
            index = landmark106to68[i]
            pt68.append(pt106[index])
        return pt68

    def extract_to_dfl_img(self, img_path, img_name, aligned_path):
        img_mat = cv2.imread(img_path)
        heigh, width, _ = img_mat.shape  # (1920, 1080, 3)
        debug_image = None
        if self.need_debug == 'yes':
            debug_image = img_mat.copy()  # debug图
            os.makedirs(self.original_folder + '/aligned_debug', exist_ok=True)

        faces = self.faceAnalysis.get(img_mat)
        if faces is None or len(faces) == 0:  # 检测是否存在脸
            return None

        img_name = img_name.replace("png", "jpg").replace("PNG", "jpg")  # 后缀变动
        i = 0
        for face in faces:
            # print(face.bbox)#xmin,ymin,xmax,ymax = box
            face_x, face_y = face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]

            image_size = self.image_size
            landmark = self.landmark106to68(face.landmark_2d_106)
            image_to_face_mat = get_transform_mat(
                landmark, image_size, self.face_style)  # 脸型,变换矩阵
            # scale_x, scale_y = 0.5, 0.5  # 7-20添加缩放变换
            # image_to_face_mat = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])  # 7-20添加缩放变换

            face_image = cv2.warpAffine(
                img_mat, image_to_face_mat, (image_size, image_size), cv2.INTER_LANCZOS4)

            output_filepath = aligned_path + "/" + img_name.split('.')[0] + '_' + str(i) + '.jpg'  # 多人

            self.cv2_imwrite(output_filepath, face_image, [int(
                cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])  # 质量

            if self.need_debug == 'yes':
                output_debug_path = self.original_folder + '/aligned_debug/' + img_name  # debug保存路径
                rect = face.bbox.astype(np.int32)  # 矩形框坐标
                draw_rect_landmarks(
                    debug_image, rect, landmark, self.face_style, image_size, transparent_mask=True)  # 画人脸框
                self.cv2_imwrite(output_debug_path, debug_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])  # 保存debug图

            dflimg = DFLJPG.load(output_filepath)
            dflimg.set_face_type(FaceType.toString(self.face_style))
            face_image_landmarks = transform_points(landmark, image_to_face_mat)
            # 相似变换之后的人脸关键点
            dflimg.set_landmarks(face_image_landmarks.tolist())
            dflimg.set_source_filename(img_name)
            dflimg.set_source_rect(face.bbox)

            # 源人脸关键点
            dflimg.set_source_landmarks(landmark)
            dflimg.set_image_to_face_mat(image_to_face_mat)
            dflimg.save()
            i += 1
            if i > int(self.face_num):
                break

    def face_det_lmk_dir(self):
        dir_path = self.original_folder
        aligned_path = self.aligned_folder
        import concurrent.futures
        # 使用多线程Thread
        with concurrent.futures.ThreadPoolExecutor(self.max_worker) as executor:
            to_do = []
            # 会读取data_src里的aligned文件夹，此时不是图片
            for img_name in os.listdir(dir_path):
                if any([img_name.lower().endswith(ext) for ext in ['jpg', 'png', 'jpeg']]):
                    try:
                        img_path = dir_path + "/" + img_name
                        future = executor.submit(
                            self.extract_to_dfl_img, img_path, img_name, aligned_path)
                        to_do.append(future)
                    except Exception as e:
                        traceback.print_exc()

            for future in tqdm.tqdm(concurrent.futures.as_completed(to_do), desc="人脸提取进度", leave=True, ascii=True,
                                    initial=0):  # 并发执行 , desc="progress", leave=True, ascii=True, initial=0
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()

    def face_det_lmk_dir2(self):  # 单线程
        dir_path = self.original_folder
        aligned_path = self.aligned_folder
        start_time = time.time()
        total_images = len(
            [img_name for img_name in os.listdir(dir_path) if img_name.lower().endswith(('jpg', 'png', 'jpeg'))])

        with tqdm.tqdm(total=total_images, desc="人脸提取进度", leave=True, initial=0, ascii=True) as pbar:
            for img_name in os.listdir(dir_path):
                if any([img_name.lower().endswith(ext) for ext in ['jpg', 'png', 'jpeg']]):
                    try:
                        img_path = dir_path + "/" + img_name
                        self.extract_to_dfl_img(img_path, img_name, aligned_path)
                        pbar.update(1)
                    except Exception as e:
                        traceback.print_exc()

        end_time = time.time()
        processing_time = end_time - start_time
        print("Processing completed in {:.2f} seconds.".format(processing_time))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default=None)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    parser.add_argument('-s', '--face_style', type=str, default='wf')
    parser.add_argument('-q', '--jpeg_quality', type=int, default=100)
    parser.add_argument('-w', '--max_worker', type=int, default=1)
    parser.add_argument('-f', '--face_num', type=int, default=6)
    parser.add_argument('-nd', '--need_debug', type=str, default='no')
    parser.add_argument('-size', '--image_size', type=int, default=512)
    parser.add_argument('-ds', '--det_size', type=int, default=640)

    args = parser.parse_args()

    # face = extract(r'E:\ui\dfl_code\workspace\data_dst', r'E:\ui\dfl_code\workspace\data_dst\aligned')
    face = extract(args.input_path, args.output_path, args.face_style, args.jpeg_quality,
                   args.face_num, args.need_debug, args.max_worker, args.image_size, args.det_size)

    face.face_det_lmk_dir()
    print("帧切脸已完成！！！")
