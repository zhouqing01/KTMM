import imageio
import os

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def main():
    orgin = '3/Ana'         #首先设置图像文件路径
    files = os.listdir(orgin)       #获取图像序列
    image_list = []
    for file in files:
        path = os.path.join(orgin, file)
        image_list.append(path)
    print(image_list)
    gif_name = 'Ana_3.gif'  #设置动态图的名字
    duration = 0.35
    create_gif(image_list, gif_name, duration)   #创建动态图


if __name__ == '__main__':
    main()