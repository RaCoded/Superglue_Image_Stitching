import cv2
import numpy as np
import torch
from utils.models.SuperGluePretrainedNetwork.models.matching import Matching
import matplotlib.pyplot as plt
import os, sys

class Superglue_Stitching():
    def __init__(self,config,min_match,device):
        self.configuration = config
        self.matcher = Matching(config).to(device).eval()
        self.min_match = min_match
        self.device = device
        self.smoothing_window_size=800
        
    def registration(self,img1,img2):
        inp1, inp2 = self.load_gray(img1,self.device), self.load_gray(img2,self.device)
        with torch.no_grad():
            pred = self.matcher({'image0': inp1, 'image1': inp2})
            kpts0, kpts1 = pred['keypoints0'][0].cpu().numpy(), pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        valid = matches > -1
        matches0, matches1 = kpts0[valid], kpts1[matches[valid]]
        self.draw_matches(img1, img2, matches0, matches1, "output/matching.jpg")
        if len(valid) > self.min_match:
            H, status = cv2.findHomography(matches1, matches0, cv2.RANSAC,5.0)
            return H
        else:
            print(" Not enought matches")
            
    def load_gray(self,img,device):
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return torch.from_numpy(img/255.).float().unsqueeze(0).unsqueeze(0).to(device)
    
    def draw_matches(self, img1, img2, kpts0, kpts1, out_path):
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        H1, W1 = img1.shape[:2]
        H2, W2 = img2.shape[:2]
        
        canvas = np.zeros((max(H1,H2), W1+W2, 3), dtype=np.uint8)
        canvas[:H1, :W1] = img1
        canvas[:H2, W1:] = img2

        canvas_rgb = canvas[..., ::-1]  

        plt.figure(figsize=(12,6))
        plt.imshow(canvas_rgb)
        
        for (x0, y0), (x1, y1) in zip(kpts0, kpts1):
            plt.plot([x0, x1 + W1], [y0, y1], color='red', lw=1)
        plt.scatter(kpts0[:,0], kpts0[:,1], c='yellow', s=5)
        plt.scatter(kpts1[:,0] + W1, kpts1[:,1], c='cyan', s=5)
        
        plt.axis('off')
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        print(f"Saved matches image to {out_path}")

        
    def create_mask(self,img1,img2,H,version):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        corners_img2 = np.array([[0,0], [w2,0], [w2,h2], [0,h2]], dtype=np.float32).reshape(-1,1,2)
        projected_corners = cv2.perspectiveTransform(corners_img2, H)
        
        x_min = max(0, int(np.min(projected_corners[:,:,0])))
        x_max = min(w1, int(np.max(projected_corners[:,:,0])))
        
        width_panorama = w1 + w2
        height_panorama = max(h1, h2)
        
        mask = np.zeros((height_panorama, width_panorama), dtype=np.float32)
        
        if version == 'left_image':
            mask[:, :x_min] = 1.0
            if x_max > x_min:
                mask[:, x_min:x_max] = np.tile(np.linspace(1, 0, x_max - x_min), (height_panorama, 1))
        else:  
            mask[:, x_max:] = 1.0
            if x_max > x_min:
                mask[:, x_min:x_max] = np.tile(np.linspace(0, 1, x_max - x_min), (height_panorama, 1))

        return cv2.merge([mask, mask, mask])
    
    def blending(self,img1,img2):
        H = self.registration(img1,img2)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        height_panorama = max(h1, h2)
        width_panorama = w1 + w2
        
        mask1 = self.create_mask(img1, img2, H, version='left_image')
        panorama1 = np.zeros((height_panorama, width_panorama, 3), dtype=np.float32)
        panorama1[0:h1, 0:w1] = img1.astype(np.float32)
        panorama1 *= mask1
        
        mask2 = self.create_mask(img1, img2, H, version='right_image')
        warped_img2 = cv2.warpPerspective(img2.astype(np.float32), H, (width_panorama, height_panorama))
        warped_img2 *= mask2
        
        result = panorama1 + warped_img2
        result = np.clip(result, 0, 255).astype(np.uint8)

        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows)+1
        min_col, max_col = min(cols), max(cols)+1
        final_result = result[min_row:max_row, min_col:max_col]

        return final_result

        
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'superpoint': {'nms_radius': 4, 'keypoint_threshold': 0.01, 'max_keypoints': -1},
        'superglue': {'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.5},
        'device': device
    }
    img1 = cv2.imread("q11.jpg")
    img2 = cv2.imread("q22.jpg")
    final= Superglue_Stitching(config,4,device).blending(img1,img2)
    cv2.imwrite('panorama.jpg', final)

if __name__ == '__main__':
    main()