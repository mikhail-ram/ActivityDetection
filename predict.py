import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from yt_dlp import YoutubeDL
from torchvision.models import mobilenet_v2
import torch.nn as nn
import torch.nn.functional as F

class MobileNetGRU(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(MobileNetGRU, self).__init__()
        self.mobilenet = mobilenet_v2(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gru = nn.GRU(input_size=1280, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.mobilenet(x)
        features = self.avgpool(features).view(batch_size, seq_len, -1)
        out, _ = self.gru(features)
        out = self.fc(out[:, -1, :])
        return out

class_names = [
    "BaseballPitch", "Basketball", "BenchPress", "Biking", "Billards", "BreastStroke", 
    "CleanAndJerk", "Diving", "Drumming", "Fencing", "GolfSwing", "HighJump", 
    "HorseRace", "HorseRiding", "HulaHoop", "JavelinThrow", "JugglingBalls", 
    "JumpingJack", "JumpRope", "Kayaking", "Lunges", "MilitaryParade", "Mixing", 
    "Nunchucks", "PizzaTossing", "PlayingGuitar", "PlayingPiano", "PlayingTabla", 
    "PlayingViolin", "PoleVault", "PommelHorse", "Pullup", "Punch", "PushUps", 
    "RockClimbingIndoor", "RopeClimbing", "Rowing", "SalsaSpin", "SkateBoarding", 
    "Skiing", "Skijet", "SoccerJuggling", "Swing", "TaiChi", "TennisSwing", 
    "ThrowDiscus", "TrampolineJumping", "VolleyballSpiking", "WalkingWithDog", "YoYo"
]

transform = Compose([Resize((224, 224)), ToTensor()])

def load_model(model_path, device):
    model = MobileNetGRU(hidden_size=256, num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def extract_frames(video_path, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    
    cap.release()
    return frames

def download_video(youtube_url, output_path="temp_video.mp4"):
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': output_path,
        'noplaylist': True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path

def predict_activity(video_input, model_path, device):
    if not os.path.isfile(video_input):
        video_path = download_video(video_input)
    else:
        video_path = video_input

    frames = extract_frames(video_path)
    frames_tensor = torch.stack([transform(frame) for frame in frames]).unsqueeze(0).to(device)
    
    model = load_model(model_path, device)
    output = model(frames_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    print(f"Predicted Activity: {class_names[predicted_class]}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "weights_dict_epoch_7.pth"
video_input = input("Enter video path or YouTube link: ")

predict_activity(video_input, model_path, device)

