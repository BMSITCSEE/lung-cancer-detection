from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_gradcam(model, input_tensor, target_class, target_layer, image_rgb):
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=True)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])
    return show_cam_on_image(image_rgb, grayscale_cam[0], use_rgb=True)
