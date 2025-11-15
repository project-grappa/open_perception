import torch
from sam2.build_sam import build_sam2_camera_predictor
import cv2

sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

cap = cv2.VideoCapture(0)

if_init = False

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        width, height = frame.shape[:2][::-1]

        if not if_init:
            predictor.load_first_frame(frame)
            if_init = True
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(0,0,
                                                        points=None,
                                                        labels=None,
                                                        bbox=[int(width/3),int(height/3),int(2*width/3),int(2*height/3)],
                                                        clear_old_points=True,
                                                        normalize_coords=True)

        else:
            out_obj_ids, out_mask_logits = predictor.track(frame)
        
        for obj_id, mask_logit in zip(out_obj_ids, out_mask_logits):
            mask = mask_logit.sigmoid().cpu().numpy()
            mask = (mask > 0.5).astype("uint8") * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()