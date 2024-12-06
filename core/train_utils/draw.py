import matplotlib.pyplot as plt
import cv2

def plot_region(polys,keypoints,img,out_path):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 创建绘图对象
    fig = plt.figure()

    # 在绘图对象上创建一个子图
    ax = fig.add_subplot(111)

    for poly in polys:

        # 将边界点分别存储在 x 和 y 列表中
        x, y = zip(*poly)

        x = list(x) + [x[0]]
        y = list(y) + [y[0]]

        # 绘制区域内部的填充
        ax.fill(x, y, color='green', alpha=0.5)

        # 绘制区域边界线
        ax.plot(x, y, color='red', linestyle='-', linewidth=2)

    # 绘制点
    # for kps in keypoints:
    #     plt.plot(kps[:, 0], kps[:, 1], 'bo')


    # 绘制原图，并设置颜色映射参数为 'None'
    ax.imshow(img)

    # 显示坐标轴
    plt.axis('off')

    # # 显示绘图结果
    # plt.show()

    # 保存图片
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)

    plt.imshow(img)
    # plt.show()
    plt.close()