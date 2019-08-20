import cv2
import numpy as np


class Circle:
    def __init__(self, img, win_name, radius, color):
        self.drawing = False
        self.img = img
        self.win_name = win_name
        self.radius = radius
        self.color = color

        cv2.namedWindow(self.win_name)
        cv2.imshow("ifft", self.img)
        self.img_fft = np.fft.fftshift(np.fft.fft2(self.img))
        fft = img.copy()
        cv2.imshow(self.win_name, cv2.normalize(20 * np.log(np.abs(self.img_fft.real)), fft, 255, 0,
                                                norm_type=cv2.NORM_MINMAX).astype(np.uint8))

    def draw(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                a = cv2.circle(self.img_fft.real, (x, y), self.radius, 1., -1)
                self.img_fft.real[:] = a.get()
                a = cv2.circle(self.img_fft.imag, (x, y), self.radius, 1., -1)
                self.img_fft.imag[:] = a.get()
                self.img = cv2.normalize(np.fft.ifft2(np.fft.ifftshift(self.img_fft)).real, self.img, 255, 0,
                                         norm_type=cv2.NORM_MINMAX)
                img = self.img_fft.real.copy()
                cv2.imshow(self.win_name, cv2.normalize(20 * np.log(np.abs(self.img_fft.real) + 1), img, 255, 0,
                                                        norm_type=cv2.NORM_MINMAX).astype(np.uint8))
                cv2.imshow("orig", self.img.astype(np.uint8))
                # cv2.imshow("thresh", cv2.adaptiveThreshold(self.img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                #                             cv2.THRESH_BINARY, 3, 2).astype(np.uint8))
                cv2.imshow("thresh", cv2.threshold(self.img.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)[1].astype(np.uint8))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False


if __name__ == '__main__':
    img = cv2.imread("/home/vstupakov/Downloads/Tile_01505_hist.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (1000, 1000))
    # img = cv2.equalizeHist(img)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                             cv2.THRESH_BINARY, 3, 2)
    cv2.imshow("orig", img.copy())
    circle = Circle(img=img, win_name="fft", radius=50, color=(0, 0, 0))
    cv2.setMouseCallback("fft", circle.draw)

    cv2.waitKey()
