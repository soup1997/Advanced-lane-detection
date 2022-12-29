# -*- coding:utf-8 -*-
import numpy as np
import cv2

class Camera:

    def __init__(self):
        self.WIDTH, self.HEIGHT = 640, 480  # 카메라 가로, 세로 크기

        # ====================
        # ROI - array 순서 : [좌하, 좌상, 우상, 우하]

        # project_video.mp4 (test.bag)을 위한 roi
        vertices = np.array([(0, 400), (self.WIDTH // 2 - 200, 330),
                                    (self.WIDTH // 2 + 200, 330), (self.WIDTH, 400)],
                                   dtype=np.int32)

        # Bird's eye View 변환을 위한 src, dst point 설정 (src 좌표에서 dst 좌표로 투시 변환)
        self.points_src = np.float32(list(vertices))
        self.points_dst = np.float32(
            [(100, self.HEIGHT), (100, 0), (self.WIDTH - 100, 0), (self.WIDTH - 100, self.HEIGHT)])
        
        # 만든 src, dst point 를 이용하여 투시 변환 행렬 생성
        self.transform_matrix = cv2.getPerspectiveTransform(self.points_src, self.points_dst)
        # 원본 영상으로 되돌리기 위한 역변환 행렬
        self.inv_transform_matrix = cv2.getPerspectiveTransform(self.points_dst, self.points_src)

    # ========================================
    # 윤곽선 검출
    # ========================================
    def canny_edge(self, img):
        # img = cv2.Canny(img, 50, 120)     # 카메라
        img = cv2.Canny(img, 25, 60)        # project_video.mp4 (test.bag)
        return img


    # ========================================
    # 흑백 영상 변환
    # ========================================
    def gray_scale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이 스케일 이미지로 변경하여 이미지 반환


    # ========================================
    # 가우시안 블러링
    # ========================================
    def gaussian_blur(self, img):
        return cv2.GaussianBlur(img, (5, 5), 0)  # 노이즈 제거(솔트 & 페퍼 노이즈) 이미지 반환


    # ========================================
    # Bird's eye View 변환
    # ========================================
    def perspective_transform(self, img):  # birds eye view
        result_img = cv2.warpPerspective(img, self.transform_matrix, (self.WIDTH, self.HEIGHT))
        return result_img  # 변환한 이미지 반환

    # ========================================
    # 영상 전처리
    # ========================================
    def pre_processing(self, img):
        img = self.gray_scale(img)
        img = self.gaussian_blur(img)
        img = self.canny_edge(img)
        img = self.perspective_transform(img)

        return img
