import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F




def convert_and_save(prediction, original_image, name, format_data, scale, threshold = 0.8, path = 'new_data/'):
    ''' function for convetr and save pictures'''

    prediction = prediction.copy()
    prediction[0]["boxes"] = prediction[0]["boxes"][prediction[0]["scores"] > threshold]
    prediction[0]["labels"] = prediction[0]["labels"][prediction[0]["scores"] > threshold]
    prediction[0]["scores"] = prediction[0]["scores"][prediction[0]["scores"] > threshold]


    prediction[0]["boxes"] = prediction[0]["boxes"][prediction[0]["labels"] == 1]

    new_boxes = prediction[0]["boxes"]

    new_b_b = []
    for i in new_boxes:
        bbx = i.detach().numpy()
        x1, y1, x2, y2 = bbx[0] / scale[0] , bbx[1] / scale[1], bbx[2] / scale[0] , bbx[3] / scale[1]
        new_b_b.append([x1, y1, x2, y2])

    new_b_b = torch.tensor(new_b_b)
    
    image_output = draw_bounding_boxes(image=original_image,
                             boxes=new_b_b,
                             #@labels=holiday_annot_labels,
                             colors="red",
                             width=2)

    img = image_output.detach()
    img = F.to_pil_image(img)
    img.save(path + 'new_' + name + '.' + format_data )
