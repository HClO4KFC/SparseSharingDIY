import multiprocessing

from PIL import Image


def open_image(file_path, type_):
    img = Image.open(file_path).convert(type_)
    img.show()


def async_img_display(file_path, type_):

    # 创建并启动进程
    process = multiprocessing.Process(target=open_image, args=(file_path, type_))
    process.start()