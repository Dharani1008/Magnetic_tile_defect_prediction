import flask
import os
from PIL import Image
import numpy as np
import random
from torchvision import datasets, transforms, models
import torch
import uuid


app = flask.Flask(__name__)

PATH = './mag-def_alexnet.pth'
checkpoint = torch.load(PATH)
model = checkpoint['model']
model

def get_predictions():

    data = {"success": False}
    task_id = str(uuid.uuid4())
    data ["id"] = task_id
    
    try:
        
        file = flask.request.files['image']
        pil_img = Image.open(file.stream)
        h, w = pil_img.height, pil_img.width
        print("{} h:{} w:{} h/w: {} mode: {}".format(task_id, h, w, h/w, pil_img.mode))
        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        img_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

        img = pil_img.convert('RGB')
        img = img_transform(img)
        img = img.numpy()
        model.to(device)
        model.eval()
           
        images = torch.from_numpy(img)
        images = images.unsqueeze(0)        
        images = images.type(torch.FloatTensor)
        images = images.to(device) # Move input tensors to the GPU/CPU
        
        output = model.forward(images)
        
        ps = torch.exp(output) # get the class probabilities from log-softmax

        pred_numpy = ps.data.cpu().numpy()
        pred_1,pred_2,pred_3,pred_4,pred_5,pred_6= pred_numpy[0, 0], pred_numpy[0, 1], pred_numpy[0, 2],pred_numpy[0,3],pred_numpy[0,4],pred_numpy[0,5]
        print('Blowhole_pred:{}, Break_pred:{}, Crack_pred:{}, Fray_pred:{}, Free_pred:{}, Uneven_pred:{}'.format(pred_1, pred_2,pred_3,pred_4,pred_5,pred_6))
        
        results = {
            'Blowhole': str(pred_1),
            'Break': str(pred_2),
            'Crack': str(pred_3),
            'Fray': str(pred_4),
            'Free': str(pred_5),
            'Uneven': str(pred_6),
        }
        

        data['predictions'] = results
        data['success'] = True
        
    except ValueError as e:
        print('ValueError')
        data['Error'] = str(e)
    except Exception as e:
        print('Exception')
        print(e)
        data['Error'] = str(e) 
        
    if "Error" in data.keys():
        print("{} {}".format(task_id, data['Error']))
    response = flask.jsonify(data)
    return response



@app.route("/api/v0", methods=["POST"])
def process_api():
    """
    cURL usage
    curl -F "image=@0_L_CC.png" {ipaddress}:5012/api/v0
    """
    return get_predictions()


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5020)

