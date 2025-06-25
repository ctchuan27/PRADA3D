import torch
import romp


if __name__ == "__main__":
    settings = romp.main.default_settings 
    settings.mode = 'webcam'
    #settings.show = True
    romp_model = romp.ROMP(settings)

    with torch.no_grad():

        cap = romp.utils.WebcamVideoStream(2)
        cap.start()
        while True:
            frame = cap.read()
            #frame = cv2.imread("./test.jpg")
            
            result = romp_model(frame)
            if result is None:
                continue
            if result["body_pose"].shape[0] > 1:
                result = {k: v[0:1] for k, v in result.items()}
            result = {
                "betas": result["smpl_betas"].mean(axis=0),
                "global_orient": result["smpl_thetas"][:, :3],
                "body_pose": result["smpl_thetas"][:, 3:],
                "trans": result["cam_trans"],
            }

        cap.stop()



