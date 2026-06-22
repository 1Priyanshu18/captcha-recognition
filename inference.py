import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T


# Vocabulary
VOCAB = ['0','1','2','3','4','5','6','7','8','9',
         'A','B','C','D','E','F','G','H','J','K',
         'M','N','P','Q','R','S','T','U','V','W','X','Y','Z']

VOCAB_SIZE = len(VOCAB)
BLANK_IDX = 0
NUM_CLASSES = VOCAB_SIZE + 1

CHAR2IDX = {c: i + 1 for i, c in enumerate(VOCAB)}
IDX2CHAR = {i + 1: c  for i, c in enumerate(VOCAB)}

# Transform
TRANSFORM = T.Compose([
    T.Grayscale(),
    T.ToTensor(),
])

# Model
class CRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, rnn_hidden=256, rnn_layers=2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if rnn_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.adaptive_pool(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        x = x.log_softmax(2)
        return x


# Beam Search
def beam_search_ctc(log_probs_seq, beam_width=20):
    beams = [('', BLANK_IDX, 0.0)]
    for t in range(len(log_probs_seq)):
        lp = log_probs_seq[t]
        new_beams = {}
        for prefix, last_char, score in beams:
            for c in range(len(lp)):
                lp_c = lp[c]
                if c == BLANK_IDX:
                    key = (prefix, BLANK_IDX)
                    new_beams[key] = max(new_beams.get(key, -1e9), score + lp_c)
                elif c == last_char:
                    key = (prefix, c)
                    new_beams[key] = max(new_beams.get(key, -1e9), score + lp_c)
                else:
                    key = (prefix + IDX2CHAR.get(c, ''), c)
                    new_beams[key] = max(new_beams.get(key, -1e9), score + lp_c)
        beams = sorted(
            [(p, lc, s) for (p, lc), s in new_beams.items()],
            key=lambda x: -x[2]
        )[:beam_width]
    return beams[0][0]


# Predictor
class CaptchaPredictor:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = CRNN(num_classes=NUM_CLASSES).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {checkpoint_path} on {self.device}")

    def predict(self, image_input, beam_width=20):
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        else:
            img = image_input.convert('RGB')

        img_t = TRANSFORM(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_probs = self.model(img_t)

        log_probs_np = log_probs.permute(1, 0, 2).cpu().numpy()[0]
        return beam_search_ctc(log_probs_np, beam_width)

    def predict_batch(self, image_inputs, beam_width=20):
        tensors = []
        for inp in image_inputs:
            img = Image.open(inp).convert('RGB') if isinstance(inp, str) else inp.convert('RGB')
            tensors.append(TRANSFORM(img))

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            log_probs = self.model(batch)

        log_probs_np = log_probs.permute(1, 0, 2).cpu().numpy()
        return [beam_search_ctc(log_probs_np[i], beam_width) for i in range(len(image_inputs))]


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python inference.py <checkpoint.pth> <image.png>")
        sys.exit(1)

    ckpt  = sys.argv[1]
    img   = sys.argv[2]

    predictor = CaptchaPredictor(ckpt)
    result    = predictor.predict(img)
    print(f"Prediction: {result}")