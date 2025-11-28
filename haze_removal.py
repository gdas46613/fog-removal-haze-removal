import PIL.Image as Image
import skimage.io as io
import numpy as np
import time
from gf import guided_filter   # assumes gf.guided_filter exists
from numba import njit
import cv2
import os
import sys

# ------------------------------
# NUMBA-OPTIMIZED HELPER FUNCTIONS
# ------------------------------

@njit
def compute_dark_channel(src, radius):
    """
    Computes the dark channel prior of the image.
    Optimized with Numba for performance.
    """
    rows, cols, _ = src.shape
    dark = np.zeros((rows, cols), dtype=np.float64)
    
    for i in range(rows):
        for j in range(cols):
            rmin = max(0, i - radius)
            rmax = min(i + radius, rows - 1)
            cmin = max(0, j - radius)
            cmax = min(j + radius, cols - 1)

            local_min = 1.0
            for rr in range(rmin, rmax + 1):
                for cc in range(cmin, cmax + 1):
                    pixel_min = src[rr, cc, 0]
                    if src[rr, cc, 1] < pixel_min:
                        pixel_min = src[rr, cc, 1]
                    if src[rr, cc, 2] < pixel_min:
                        pixel_min = src[rr, cc, 2]
                    if pixel_min < local_min:
                        local_min = pixel_min
            dark[i, j] = local_min
    return dark


@njit
def compute_transmission(src, Alight, radius, omega):
    """
    Computes the initial transmission map.
    Alight should be a length-3 array (RGB).
    """
    rows, cols, _ = src.shape
    tran = np.zeros((rows, cols), dtype=np.float64)
    
    # Prevent division by zero in case Alight has zeros
    A0 = max(Alight[0], 1e-6)
    A1 = max(Alight[1], 1e-6)
    A2 = max(Alight[2], 1e-6)
    
    for i in range(rows):
        for j in range(cols):
            rmin = max(0, i - radius)
            rmax = min(i + radius, rows - 1)
            cmin = max(0, j - radius)
            cmax = min(j + radius, cols - 1)

            local_min = 1.0
            for rr in range(rmin, rmax + 1):
                for cc in range(cmin, cmax + 1):
                    pixel_min = src[rr, cc, 0] / A0
                    val1 = src[rr, cc, 1] / A1
                    if val1 < pixel_min:
                        pixel_min = val1
                    val2 = src[rr, cc, 2] / A2
                    if val2 < pixel_min:
                        pixel_min = val2
                    if pixel_min < local_min:
                        local_min = pixel_min
            tran[i, j] = 1.0 - omega * local_min
    return tran


# ------------------------------
# MAIN CLASS
# ------------------------------

class HazeRemoval(object):
    def __init__(self, omega=0.95, t0=0.1, radius=7, r=20, eps=0.001):
        self.omega = omega
        self.t0 = t0
        self.radius = radius
        self.r = r
        self.eps = eps
        # Watermark Text Configuration
        self.watermark_text = "Made by Loknath Das ECE 4th Year"
        # placeholders
        self.src = None
        self.dark = None
        self.Alight = None
        self.tran = None
        self.gtran = None
        self.dst = None
        self.file_name = "output.jpg"

    def open_image(self, img_path):
        self.file_name = os.path.basename(img_path)
        img = Image.open(img_path).convert("RGB")
        self.src = np.array(img).astype(np.float64) / 255.0
        self.rows, self.cols, _ = self.src.shape
        print(f"Opened image '{img_path}' ({self.rows}x{self.cols})")

    def get_dark_channel(self):
        print("Computing dark channel prior...")
        start = time.time()
        self.dark = compute_dark_channel(self.src, self.radius)
        print(f"Dark channel computed in {time.time() - start:.3f}s")

    def get_air_light(self):
        print("Estimating air light...")
        start = time.time()
        flat = self.dark.flatten()
        flat.sort()
        num = max(1, int(self.rows * self.cols * 0.001))  # top 0.1% pixels at least 1
        threshold = flat[-num]
        # Select pixels where dark channel >= threshold
        mask = self.dark >= threshold
        tmp = self.src[mask]
        if tmp.size == 0:
            # fallback: use brightest pixels in the image
            reshaped = self.src.reshape(-1, 3)
            # take mean of top num brightest by intensity
            intensity = reshaped.mean(axis=1)
            idx = np.argsort(intensity)
            top = reshaped[idx[-num:]]
            self.Alight = top.mean(axis=0)
        else:
            # tmp has shape (k,3); compute mean of the brightest num rows by intensity
            if tmp.shape[0] <= num:
                self.Alight = tmp.mean(axis=0)
            else:
                intens = tmp.mean(axis=1)
                idx = np.argsort(intens)
                top = tmp[idx[-num:]]
                self.Alight = top.mean(axis=0)
        print("Airlight =", self.Alight)
        print(f"Air light computed in {time.time() - start:.3f}s")

    def get_transmission(self):
        print("Computing transmission map...")
        start = time.time()
        self.tran = compute_transmission(self.src, self.Alight, self.radius, self.omega)
        # clamp to [0,1]
        self.tran = np.clip(self.tran, 0.0, 1.0)
        print(f"Transmission computed in {time.time() - start:.3f}s")

    def refine_transmission(self):
        """
        Refine transmission using the imported guided_filter.
        The guided_filter is expected to accept (guidance, src_trans, r, eps) or similar.
        Keep the call consistent with your gf.guided_filter implementation.
        """
        print("Refining transmission using guided filter...")
        start = time.time()
        # The imported guided_filter likely expects guidance image and the filtering map.
        # we pass src (float RGB) and tran (float gray). If your guided_filter expects other shapes or types adjust accordingly.
        self.gtran = guided_filter(self.src, self.tran, self.r, self.eps)
        # Ensure gtran is in [0,1]
        self.gtran = np.clip(self.gtran, 0.0, 1.0)
        print(f"Guided filtering done in {time.time() - start:.3f}s")

    def recover(self):
        print("Recovering final dehazed image...")
        start = time.time()
        t = np.maximum(self.gtran, self.t0)
        t = np.repeat(t[:, :, np.newaxis], 3, axis=2)
        Alight_reshaped = self.Alight.reshape((1, 1, 3))
        self.dst = ((self.src - Alight_reshaped) / t + Alight_reshaped) * 255.0
        self.dst = np.clip(self.dst, 0, 255).astype(np.uint8)
        print(f"Image recovered in {time.time() - start:.3f}s")

    def add_watermark(self, image_data):
        """
        Adds watermark text to the image.
        Expects image_data in BGR format (uint8) OR a single-channel uint8 image.
        """
        img_out = image_data.copy()
        
        # If grayscale, convert to BGR
        if len(img_out.shape) == 2:
            img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)
        
        h, w = img_out.shape[:2]
        font_scale = max(0.5, w / 1000.0)
        thickness = max(1, int(font_scale * 2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        (text_w, text_h), baseline = cv2.getTextSize(self.watermark_text, font, font_scale, thickness)
        x = w - text_w - 10
        y = h - 10
        
        yellow = (0, 255, 255)
        black = (0, 0, 0)
        
        cv2.putText(img_out, self.watermark_text, (x+2, y+2), font, font_scale, black, thickness+2, cv2.LINE_AA)
        cv2.putText(img_out, self.watermark_text, (x, y), font, font_scale, yellow, thickness, cv2.LINE_AA)
        
        return img_out

    def show(self):
        print("Saving output images...")
        if not os.path.exists("img"):
            os.makedirs("img")

        # 1. Source Image (Float RGB -> Uint8 BGR)
        src_uint = (self.src * 255).astype(np.uint8)
        src_bgr = src_uint[:, :, ::-1] # RGB -> BGR
        cv2.imwrite("img/src.jpg", self.add_watermark(src_bgr))

        # 2. Dark Channel (Float Gray -> Uint8 Gray)
        dark_uint = (self.dark * 255).astype(np.uint8)
        cv2.imwrite("img/dark.jpg", self.add_watermark(dark_uint))

        # 3. Transmission Map (Float Gray -> Uint8 Gray)
        tran_uint = (self.tran * 255).astype(np.uint8)
        cv2.imwrite("img/tran.jpg", self.add_watermark(tran_uint))

        # 4. Guided Transmission (Float Gray -> Uint8 Gray)
        gtran_uint = (self.gtran * 255).astype(np.uint8)
        cv2.imwrite("img/gtran.jpg", self.add_watermark(gtran_uint))

        # 5. Final Result (Uint8 RGB -> Uint8 BGR)
        dst_bgr = self.dst[:, :, ::-1] # RGB -> BGR
        cv2.imwrite("img/dst.jpg", self.add_watermark(dst_bgr))
        
        # Save a copy as dehazed_<originalname> (skimage expects RGB)
        final_watermarked_bgr = self.add_watermark(dst_bgr)
        final_watermarked_rgb = final_watermarked_bgr[:, :, ::-1]
        
        output_filename = f"dehazed_{self.file_name}"
        io.imsave(output_filename, final_watermarked_rgb)
        print(f"Processing Complete. Saved result to 'img/dst.jpg' and '{output_filename}'")


# ------------------------------
# MAIN EXECUTION
# ------------------------------

def main():
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Error: No input image provided.")
        print("Usage: python haze_removal.py <image_path>")
        print("Example: python haze_removal.py myphoto.jpg")
        return

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' was not found.")
        return

    try:
        hr = HazeRemoval()
        hr.open_image(input_path)
        hr.get_dark_channel()
        hr.get_air_light()
        hr.get_transmission()
        hr.refine_transmission()
        hr.recover()
        hr.show()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()