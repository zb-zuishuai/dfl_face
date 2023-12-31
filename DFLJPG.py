import pickle
import traceback
import numpy as np
import cv2
import struct
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


def struct_unpack(data, counter, fmt):
    fmt_size = struct.calcsize(fmt)
    return (counter + fmt_size,) + struct.unpack(fmt, data[counter:counter + fmt_size])


def cv2_imread(filename, flags=cv2.IMREAD_UNCHANGED, loader_func=None, verbose=True):
    """
    allows to open non-english characters path
    """
    try:
        if loader_func is not None:
            bytes = bytearray(loader_func(filename))
        else:
            with open(filename, "rb") as stream:
                bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        return cv2.imdecode(numpyarray, flags)
    except:
        # if verbose:
        #     io.log_err(f"Exception occured in cv2_imread : {traceback.format_exc()}")
        return None


class DFLJPG(object):
    def __init__(self, filename):
        self.filename = filename
        self.data = b""
        self.length = 0
        self.chunks = []
        self.dfl_dict = None
        self.shape = None
        self.img = None

    @staticmethod
    def load_raw(filename, loader_func=None):
        try:
            if loader_func is not None:
                data = loader_func(filename)
            else:
                with open(filename, "rb") as f:
                    data = f.read()
        except:
            raise FileNotFoundError(filename)

        try:
            inst = DFLJPG(filename)
            inst.data = data
            inst.length = len(data)
            inst_length = inst.length
            chunks = []
            data_counter = 0
            while data_counter < inst_length:
                chunk_m_l, chunk_m_h = struct.unpack("BB", data[data_counter:data_counter + 2])
                data_counter += 2

                if chunk_m_l != 0xFF:
                    raise ValueError(f"No Valid JPG info in {filename}")

                chunk_name = None
                chunk_size = None
                chunk_data = None
                chunk_ex_data = None
                is_unk_chunk = False

                if chunk_m_h & 0xF0 == 0xD0:
                    n = chunk_m_h & 0x0F

                    if n >= 0 and n <= 7:
                        chunk_name = "RST%d" % (n)
                        chunk_size = 0
                    elif n == 0x8:
                        chunk_name = "SOI"
                        chunk_size = 0
                        if len(chunks) != 0:
                            raise Exception("")
                    elif n == 0x9:
                        chunk_name = "EOI"
                        chunk_size = 0
                    elif n == 0xA:
                        chunk_name = "SOS"
                    elif n == 0xB:
                        chunk_name = "DQT"
                    elif n == 0xD:
                        chunk_name = "DRI"
                        chunk_size = 2
                    else:
                        is_unk_chunk = True
                elif chunk_m_h & 0xF0 == 0xC0:
                    n = chunk_m_h & 0x0F
                    if n == 0:
                        chunk_name = "SOF0"
                    elif n == 2:
                        chunk_name = "SOF2"
                    elif n == 4:
                        chunk_name = "DHT"
                    else:
                        is_unk_chunk = True
                elif chunk_m_h & 0xF0 == 0xE0:
                    n = chunk_m_h & 0x0F
                    chunk_name = "APP%d" % (n)
                else:
                    is_unk_chunk = True

                # if is_unk_chunk:
                #    #raise ValueError(f"Unknown chunk {chunk_m_h} in {filename}")
                #    io.log_info(f"Unknown chunk {chunk_m_h} in {filename}")

                if chunk_size == None:  # variable size
                    chunk_size, = struct.unpack(">H", data[data_counter:data_counter + 2])
                    chunk_size -= 2
                    data_counter += 2

                if chunk_size > 0:
                    chunk_data = data[data_counter:data_counter + chunk_size]
                    data_counter += chunk_size

                if chunk_name == "SOS":
                    c = data_counter
                    while c < inst_length and (data[c] != 0xFF or data[c + 1] != 0xD9):
                        c += 1

                    chunk_ex_data = data[data_counter:c]
                    data_counter = c

                chunks.append({'name': chunk_name,
                               'm_h': chunk_m_h,
                               'data': chunk_data,
                               'ex_data': chunk_ex_data,
                               })
            inst.chunks = chunks

            return inst
        except Exception as e:
            raise Exception(f"Corrupted JPG file {filename} {e}")

    @staticmethod
    def load(filename, loader_func=None):
        try:
            inst = DFLJPG.load_raw(filename, loader_func=loader_func)
            inst.dfl_dict = {}

            for chunk in inst.chunks:
                if chunk['name'] == 'APP0':
                    d, c = chunk['data'], 0
                    c, id, _ = struct_unpack(d, c, "=4sB")

                    if id == b"JFIF":
                        c, ver_major, ver_minor, units, Xdensity, Ydensity, Xthumbnail, Ythumbnail = struct_unpack(d, c,
                                                                                                                   "=BBBHHBB")
                    else:
                        raise Exception("Unknown jpeg ID: %s" % (id))
                elif chunk['name'] == 'SOF0' or chunk['name'] == 'SOF2':
                    d, c = chunk['data'], 0
                    c, precision, height, width = struct_unpack(d, c, ">BHH")
                    inst.shape = (height, width, 3)

                elif chunk['name'] == 'APP15':
                    if type(chunk['data']) == bytes:
                        inst.dfl_dict = pickle.loads(chunk['data'])

            return inst
        except Exception as e:
            print(f'Exception occured while DFLJPG.load : {traceback.format_exc()}')
            return None

    def has_data(self):
        return len(self.dfl_dict.keys()) != 0

    def save(self):
        try:
            with open(self.filename, "wb") as f:
                f.write(self.dump())
        except:
            raise Exception(f'cannot save {self.filename}')

    def dump(self):
        data = b""

        dict_data = self.dfl_dict

        # Remove None keys
        for key in list(dict_data.keys()):
            if dict_data[key] is None:
                dict_data.pop(key)

        for chunk in self.chunks:
            if chunk['name'] == 'APP15':
                self.chunks.remove(chunk)
                break

        last_app_chunk = 0
        for i, chunk in enumerate(self.chunks):
            if chunk['m_h'] & 0xF0 == 0xE0:
                last_app_chunk = i

        dflchunk = {'name': 'APP15',
                    'm_h': 0xEF,
                    'data': pickle.dumps(dict_data),
                    'ex_data': None,
                    }
        self.chunks.insert(last_app_chunk + 1, dflchunk)

        for chunk in self.chunks:
            data += struct.pack("BB", 0xFF, chunk['m_h'])
            chunk_data = chunk['data']
            if chunk_data is not None:
                data += struct.pack(">H", len(chunk_data) + 2)
                data += chunk_data

            chunk_ex_data = chunk['ex_data']
            if chunk_ex_data is not None:
                data += chunk_ex_data

        return data

    def get_face_type(self):
        return self.dfl_dict.get('face_type', FaceType.toString(FaceType.FULL))

    def set_face_type(self, face_type):
        self.dfl_dict['face_type'] = face_type

    def get_landmarks(self):
        return np.array(self.dfl_dict['landmarks'])

    def set_landmarks(self, landmarks):
        self.dfl_dict['landmarks'] = landmarks

    def get_eyebrows_expand_mod(self):
        return self.dfl_dict.get('eyebrows_expand_mod', 1.0)

    def set_eyebrows_expand_mod(self, eyebrows_expand_mod):
        self.dfl_dict['eyebrows_expand_mod'] = eyebrows_expand_mod

    def get_source_filename(self):
        return self.dfl_dict.get('source_filename', None)

    def set_source_filename(self, source_filename):
        self.dfl_dict['source_filename'] = source_filename

    def get_source_rect(self):
        return self.dfl_dict.get('source_rect', None)

    def set_source_rect(self, source_rect):
        self.dfl_dict['source_rect'] = source_rect

    def get_source_landmarks(self):
        return np.array(self.dfl_dict.get('source_landmarks', None))

    def set_source_landmarks(self, source_landmarks):
        self.dfl_dict['source_landmarks'] = source_landmarks

    def set_image_to_face_mat(self, image_to_face_mat):
        self.dfl_dict['image_to_face_mat'] = image_to_face_mat

    def get_dict(self):
        return self.dfl_dict

    def set_dict(self, dict_data=None):
        self.dfl_dict = dict_data

    def has_xseg_mask(self):
        return self.dfl_dict.get('xseg_mask', None) is not None

    def get_xseg_mask_compressed(self):
        mask_buf = self.dfl_dict.get('xseg_mask', None)
        if mask_buf is None:
            return None

        return mask_buf

    def get_xseg_mask(self):
        mask_buf = self.dfl_dict.get('xseg_mask', None)
        # 首先从'dfl_dict'字典中获取"xseg_mask"键对应的值，并将其存储在变量"mask_buf"中。
        # 如果"xseg_mask"为空，则返回None，表示没有可用的分割遮罩数据。
        if mask_buf is None:
            return None

        # 使用OpenCV库中的cv2.imdecode()函数对分割遮罩数据进行解码，并将结果存储在变量"img"中。
        img = cv2.imdecode(mask_buf, cv2.IMREAD_UNCHANGED)

        # 检查图像的维度是否为二维（即灰度图像），如果是，则在最后一个维度上添加新维度，将其转换为三维图像数据。
        # 这是为了使其与其他彩色图像数据保持一致。
        if len(img.shape) == 2:
            img = img[..., None]

        # 返回归一化后的浮点数类型数组，其范围在[0, 1]之间。这里除以255是为了将像素值缩放到0-1范围内的浮点数。
        return img.astype(np.float32) / 255.0

    def get_img(self):
        if self.img is None:
            self.img = cv2_imread(self.filename)
        return self.img
