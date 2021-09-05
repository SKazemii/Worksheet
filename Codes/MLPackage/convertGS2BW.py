
# from cv2 
import cv2
import numpy as np

class convertGS2BW:
    mode = ""
    TH = 0
    
    def __init__(self, mode, TH = 0):
        self.mode = mode
        if  not TH == 0:
            self.TH = TH

    def GS2BW(self, img):
        if self.mode == "simple":
            threshold, BW = cv2.threshold(img, self.TH, 255, cv2.THRESH_BINARY)
            return BW, threshold
        elif self.mode == "otsu":
            threshold, BW = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return BW, threshold
        elif self.mode == "adaptive":
            BW = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
            return BW, "adaptive"

    
    #defaultBinarize = 'simple'; % 'otsu', 'adaptive'

def main():
    a = convertGS2BW(mode = "adaptive", TH = 1.0)
    # image1 = cv2.imread('input1.jpg')
    i3D = np.load("./Codes/Worksheet01/ToonCodes/3D.npy")

    BW, threshold = a.GS2BW(i3D[:,:,40])

    # np.array(BW)
    print(type(BW))
    print(threshold)
    cv2.imshow('Binary Threshold', BW)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()   
