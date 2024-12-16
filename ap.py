!pip install tensorflow opencv-python numpy facenet-pytorch mtcnn torch torchvision
!git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git

import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image


class LivenessDetector:
    def __init__(self):
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Setup Silent-Face-Anti-Spoofing
        self.silent_face_model = self.load_silent_face_model()

        # Initialize transform for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def load_silent_face_model(self): pass
      # model_path = "Silent-Face-Anti-Spoofing/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"

      # model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
      # model.eval()  # Set the model to evaluation mode
      # return model

    def detect_face(self, image):
        """Detect faces in image using MTCNN"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        boxes, _ = self.mtcnn.detect(image)
        return boxes

    def check_liveness(self, image):
        """
        Check if detected face is live or spoof
        Returns: score between 0 (spoof) and 1 (real)
        """
        boxes = self.detect_face(image)

        if boxes is None:
            return 0.0

        # Process each detected face
        results = []
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            face = image[y1:y2, x1:x2]

            # Preprocess face
            face = cv2.resize(face, (80, 80))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = Image.fromarray(face)
            face = self.transform(face).unsqueeze(0)

            # Get prediction
            # This is where you'd run the actual model inference
            score = 0.7
            results.append(score)

        return np.mean(results) if results else 0.0

# Example usage
def demo_liveness_detection():
    # Initialize detector
    detector = LivenessDetector()

    # Setup video capture (0 for webcam)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check liveness
        score = detector.check_liveness(frame)

        # Draw result on frame
        text = f"Liveness: {score:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display result
        cv2.imshow('Liveness Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# To run the demo in Colab, you'll need to modify the display part
# Here's how to display in Colab:
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode, b64encode

def colab_webcam():
    js = Javascript('''
        async function captureFrame() {
            const div = document.createElement('div');
            document.body.appendChild(div);
            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            const dataUrl = canvas.toDataURL('image/jpeg');

            stream.getVideoTracks()[0].stop();
            div.remove();
            return dataUrl
        }
        ''')
    display(js)
    data_url = eval_js('captureFrame()')
    binary = b64decode(data_url.split(',')[1])
    return cv2.imdecode(np.frombuffer(binary, np.uint8), cv2.IMREAD_COLOR)

# To use in Colab:
detector = LivenessDetector()
frame = colab_webcam()
score = detector.check_liveness(frame)
print(f"Liveness score: {score:.2f}")

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class MiniFASNetV2(nn.Module):
    def __init__(self):
        super(MiniFASNetV2, self).__init__()

        # First conv layer
        self.conv1 = ConvBlock(3, 64, kernel_size=3, stride=2, padding=1)

        # Depthwise conv
        self.conv2_dw = DepthwiseConvBlock(64, 64, kernel_size=3, stride=1, padding=1)

        # Conv block with residual
        self.conv_23 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            DepthwiseConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 64, kernel_size=1, stride=1, padding=0)
        )

        # Additional layers
        self.conv_3 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            DepthwiseConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 64, kernel_size=1, stride=1, padding=0)
        )

        # Final layers
        self.conv_4 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(128 * 5 * 5, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)

class LivenessDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = MiniFASNetV2().to(self.device)

        # Load pretrained weights
        weights_path = "./Silent-Face-Anti-Spoofing/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            # Remove 'module.' prefix from state dict keys if present
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        else:
            raise FileNotFoundError("Model weights not found. Please download them first.")

        self.model.eval()

        # Initialize transform for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((80, 80)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces

    def preprocess_face(self, face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = Image.fromarray(face)
        face = self.transform(face)
        return face.unsqueeze(0)

    def check_liveness(self, image):
        faces = self.detect_face(image)

        if len(faces) == 0:
            return 0.0

        results = []
        for (x, y, w, h) in faces:
            # Extract and preprocess face
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, (80, 80))
            face_tensor = self.preprocess_face(face)

            # Get prediction
            with torch.no_grad():
                score = self.model(face_tensor.to(self.device))
                score = score.cpu().numpy()[0][0]
            results.append(score)

            # Draw rectangle and score
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f'Score: {score:.2f}', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return np.mean(results)

# For Google Colab usage
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode, b64encode

def colab_webcam():
    js = Javascript('''
        async function captureFrame() {
            const div = document.createElement('div');
            document.body.appendChild(div);
            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            const dataUrl = canvas.toDataURL('image/jpeg');

            stream.getVideoTracks()[0].stop();
            div.remove();
            return dataUrl
        }
        ''')
    display(js)
    data_url = eval_js('captureFrame()')
    binary = b64decode(data_url.split(',')[1])
    return cv2.imdecode(np.frombuffer(binary, np.uint8), cv2.IMREAD_COLOR)

# Usage example
def run_demo():
    print("Initializing detector...")
    detector = LivenessDetector()
    print("Taking a photo...")
    frame = colab_webcam()
    print("Analyzing liveness...")
    score = detector.check_liveness(frame)
    print(f"Liveness score: {score:.2f}")

    # Display the result
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# To use, simply run:
# run_demo()

#INCEPTION RESNET , MTCNN

def colab_webcam():
    js = Javascript('''
        async function captureFrame() {
            const div = document.createElement('div');
            document.body.appendChild(div);
            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            const dataUrl = canvas.toDataURL('image/jpeg');

            stream.getVideoTracks()[0].stop();
            div.remove();
            return dataUrl
        }
        ''')
    display(js)
    data_url = eval_js('captureFrame()')
    binary = b64decode(data_url.split(',')[1])
    return cv2.imdecode(np.frombuffer(binary, np.uint8), cv2.IMREAD_COLOR)

# Import required libraries
import cv2
import numpy as np
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms
from IPython.display import display, Javascript, HTML, clear_output
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import time

# Check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class SimpleDepthModel(nn.Module):
    def __init__(self):
        super(SimpleDepthModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 40 * 40, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def get_transform():
    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

class FaceLivenessDetector:
    def __init__(self):
        self.mtcnn = MTCNN(margin=20, keep_all=True, device=device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.depth_model = SimpleDepthModel().to(device)
        self.transform = get_transform()

    def detect_face(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Detect faces
        boxes, _ = self.mtcnn.detect(frame_pil)

        if boxes is None:
            return None
        return boxes.astype(int)

    def get_face_depth(self, face_tensor):
        with torch.no_grad():
            depth_score = self.depth_model(face_tensor)
        return depth_score.item()

    def get_liveness_score(self, face_tensor):
        with torch.no_grad():
            embedding = self.resnet(face_tensor)
            score = torch.norm(embedding).item()
        return score

    def process_frame(self, frame):
        boxes = self.detect_face(frame)
        if boxes is None:
            return frame, "No face detected", None

        results = []
        for box in boxes:
            x1, y1, x2, y2 = box

            # Extract and process face
            face_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_tensor = self.transform(face_pil).unsqueeze(0).to(device)

            # Get scores
            depth_score = self.get_face_depth(face_tensor)
            liveness_score = self.get_liveness_score(face_tensor)

            # Determine if face is real
            is_real = liveness_score < 1.5 and depth_score > 0.5

            # Draw box and scores
            color = (0, 255, 0) if is_real else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            status = "Real" if is_real else "Fake"
            cv2.putText(frame, f"{status} Face", (x1, y1-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Depth: {depth_score:.2f}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            results.append({
                'is_real': is_real,
                'depth_score': depth_score,
                'liveness_score': liveness_score
            })

        status = "Live Face Detected" if any(r['is_real'] for r in results) else "Fake Face Detected"
        return frame, status, results

def js_to_image(js_reply):
    image_bytes = b64decode(js_reply.split(',')[1])
    jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(jpg_as_np, flags=1)

def bbox_to_bytes(bbox_array):
    ret, png = cv2.imencode('.png', bbox_array)
    return b64encode(png.tobytes()).decode('utf-8')

# JavaScript code for video streaming
js_code = """
    var video;
    var div = null;
    var stream;
    var captureCanvas;
    var imgElement;
    var labelElement;

    var pendingResolve = null;
    var shutdown = false;

    function removeDom() {
       stream.getVideoTracks()[0].stop();
       video.remove();
       div.remove();
       video = null;
       div = null;
       stream = null;
       imgElement = null;
       captureCanvas = null;
       labelElement = null;
    }

    function onAnimationFrame() {
      if (!shutdown) {
        window.requestAnimationFrame(onAnimationFrame);
      }
      if (pendingResolve) {
        var result = "";
        if (!shutdown) {
          captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
          result = captureCanvas.toDataURL();
        }
        pendingResolve(result);
        pendingResolve = null;
      }
    }

    async function createDom() {
      if (div !== null) {
        return stream;
      }
      div = document.createElement('div');
      div.style.border = '2px solid black';
      div.style.padding = '3px';
      div.style.width = '100%';
      div.style.maxWidth = '600px';
      document.body.appendChild(div);

      const modelOut = document.createElement('div');
      modelOut.innerHTML = "<span>Status:</span>";
      labelElement = document.createElement('span');
      labelElement.innerText = 'No faces detected';
      modelOut.appendChild(labelElement);
      div.appendChild(modelOut);

      video = document.createElement('video');
      video.style.display = 'block';
      video.width = div.clientWidth - 6;
      video.setAttribute('playsinline', '');
      video.onclick = () => { shutdown = true; };
      stream = await navigator.mediaDevices.getUserMedia(
          {video: { facingMode: "user" }});
      div.appendChild(video);

      imgElement = document.createElement('img');
      imgElement.style.position = 'absolute';
      imgElement.style.zIndex = 1;
      imgElement.onclick = () => { shutdown = true; };
      div.appendChild(imgElement);

      const instruction = document.createElement('div');
      instruction.innerHTML =
          '<span style="color: red; font-weight: bold;">' +
          'Click here to stop camera</span>';
      div.appendChild(instruction);
      instruction.onclick = () => { shutdown = true; };

      video.srcObject = stream;
      await video.play();

      captureCanvas = document.createElement('canvas');
      captureCanvas.width = 640;
      captureCanvas.height = 480;
      window.requestAnimationFrame(onAnimationFrame);

      return stream;
    }
    async function stream_frame() {
      if (shutdown) {
        removeDom();
        shutdown = false;
        return '';
      }

      if (pendingResolve) {
        throw new Error('previous frame not finished');
      }

      return new Promise(function(resolve, reject) {
        pendingResolve = resolve;
      });
    }
    """

def start_video():
    js = Javascript(js_code)
    display(js)

def process_video_frame(label, frame):
    data = eval_js('stream_frame()')
    if data == '':
        return None

    png_bytes = bbox_to_bytes(frame)

    eval_js('document.querySelector("span").innerText = "{}"'.format(label))
    eval_js('document.querySelector("img").src = "data:image/png;base64,{}"'.format(png_bytes))
    return True

def main():
    try:
        # Initialize detector
        print("Initializing face liveness detector...")
        detector = FaceLivenessDetector()

        # Start video
        print("Starting video stream...")
        start_video()

        print("Processing frames...")
        while True:
            # Get frame from webcam
            js_reply = eval_js('stream_frame()')
            if not js_reply:
                break

            # Convert to OpenCV Image
            frame = js_to_image(js_reply)

            # Process frame
            processed_frame, status, results = detector.process_frame(frame)

            # Display results
            if results:
                for result in results:
                    print(f"Depth Score: {result['depth_score']:.2f}, "
                          f"Liveness Score: {result['liveness_score']:.2f}, "
                          f"Status: {'Real' if result['is_real'] else 'Fake'}")

            # Update display
            if not process_video_frame(status, processed_frame):
                break

            # Small delay to prevent overwhelming the browser
            time.sleep(0.1)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        print("Stopping video stream...")

if __name__ == "__main__":
    # Install required packages if not already installed
    !pip install facenet-pytorch opencv-python-headless

    # Run the main function
    main()

from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import cv2
from PIL import Image

model = InceptionResnetV1(pretrained='vggface2').eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


summary(model, input_size=(3, 160, 160))


#RESNET

import torch
import torchvision.models as models
from torchsummary import summary

resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

summary(resnet, input_size=(3, 224, 224))


import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN

class LivenessDetector:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize ResNet50 model
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)

        # Load trained weights if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading model weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("WARNING: No model weights provided. The model will not make meaningful predictions!")

        self.model.eval().to(self.device)

        # Initialize transform for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

        # Initialize face detector with more lenient parameters
        self.face_detector = MTCNN(
            keep_all=True,
            device=self.device,
            thresholds=[0.6, 0.7, 0.7],  # More lenient thresholds
            min_face_size=60  # Smaller minimum face size
        )

    def detect_face(self, image):
        """Detect faces with debug information"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                if isinstance(image, np.ndarray):
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image

            boxes, _ = self.face_detector.detect(image_rgb)

            if boxes is None:
                print("No faces detected in the image")
                return None

            print(f"Detected {len(boxes)} faces")
            return boxes

        except Exception as e:
            print(f"Error in face detection: {str(e)}")
            return None

    def preprocess_face(self, face):
        """Preprocess face with debug information"""
        try:
            if face.shape[0] == 0 or face.shape[1] == 0:
                print("Invalid face crop dimensions")
                return None

            face_pil = Image.fromarray(face)
            face_tensor = self.transform(face_pil)
            return face_tensor.unsqueeze(0)

        except Exception as e:
            print(f"Error in face preprocessing: {str(e)}")
            return None

    def check_liveness(self, image):
        """Check liveness with detailed debugging"""
        faces = self.detect_face(image)

        if faces is None or len(faces) == 0:
            print("No faces detected for liveness check")
            return 0.0

        results = []
        for i, box in enumerate(faces):
            try:
                x1, y1, x2, y2 = map(int, box)

                # Ensure valid coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

                # Extract and preprocess face
                face = image[y1:y2, x1:x2]

                if face.size == 0:
                    print(f"Invalid face crop for face {i+1}")
                    continue

                face_tensor = self.preprocess_face(face)
                if face_tensor is None:
                    continue

                # Get prediction
                with torch.no_grad():
                    output = self.model(face_tensor.to(self.device))
                    probabilities = torch.softmax(output, dim=1)
                    score = probabilities[0][1].item()
                    print(f"Face {i+1} liveness score: {score:.3f}")
                    results.append(score)

                # Draw rectangle and score
                color = (0, int(255 * score), 0)  # Color based on score
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f'Score: {score:.3f}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            except Exception as e:
                print(f"Error processing face {i+1}: {str(e)}")
                continue

        if not results:
            print("No valid liveness scores computed")
            return 0.0

        mean_score = np.mean(results)
        print(f"Mean liveness score: {mean_score:.3f}")
        return mean_score

def run_demo(model_path=None):
    print("Initializing detector...")
    detector = LivenessDetector(model_path)

    print("\nTaking a photo...")
    frame = colab_webcam()

    if frame is None:
        print("Failed to capture image")
        return

    print(f"\nImage shape: {frame.shape}")
    print("Analyzing liveness...")

    score = detector.check_liveness(frame)
    print(f"\nFinal liveness score: {score:.3f}")

    # Display the result
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Usage example:
run_demo()

import math
import time
import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
from google.colab.patches import cv2_imshow

# Function to capture frames from the webcam in Colab
def colab_webcam():
    js = Javascript('''
        async function captureFrame() {
            const div = document.createElement('div');
            document.body.appendChild(div);
            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            const dataUrl = canvas.toDataURL('image/jpeg');

            stream.getVideoTracks()[0].stop();
            div.remove();
            return dataUrl
        }
        ''')
    display(js)
    data_url = eval_js('captureFrame()')
    binary = b64decode(data_url.split(',')[1])
    return cv2.imdecode(np.frombuffer(binary, np.uint8), cv2.IMREAD_COLOR)
# Load the YOLO model
model = YOLO("l_version_1_300.pt")

classNames = ["fake", "real"]
confidence = 0.6

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()

    # Capture frame from webcam
    img = colab_webcam()

    if img is None:
        break  # Exit if no image is captured

    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            if conf > confidence:
              print(f'Detected: {classNames[cls]} with confidence: {conf}')
              if classNames[cls] == 'fake':
                color = (0, 255, 0)  # Green for 'fake'
              else:
                  color = (0, 0, 255)  # Red for 'real'

              cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
              cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                   colorB=color)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    # Display the image with detections
    cv2_imshow(img)
    cv2_waitKey(1)

!pip install ultralytics cvzone

import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import torch.nn as nn
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

# Function to capture an image from the webcam in Google Colab
def colab_webcam():
    js = Javascript('''
        async function captureFrame() {
            const div = document.createElement('div');
            document.body.appendChild(div);
            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            const dataUrl = canvas.toDataURL('image/jpeg');

            stream.getVideoTracks()[0].stop();
            div.remove();
            return dataUrl
        }
        ''')
    display(js)
    data_url = eval_js('captureFrame()')
    binary = b64decode(data_url.split(',')[1])
    return cv2.imdecode(np.frombuffer(binary, np.uint8), cv2.IMREAD_COLOR)

# Model for Depth Analysis - Placeholder for simplicity
class DepthAnalysisModel(nn.Module):
    def __init__(self):
        super(DepthAnalysisModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 64 * 64, 1)  # Assuming a 64x64 image size for depth

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        depth_output = self.fc(x)
        return depth_output

# Texture analysis using LBP (Local Binary Patterns)
def texture_analysis(face_crop):
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    lbp = cv2.calcHist([gray], [0], None, [256], [0, 256])
    lbp = lbp / np.sum(lbp)  # Normalize histogram
    return lbp

class LivenessDetector:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize ResNet50 model
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

        # Load trained weights if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading model weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("WARNING: No model weights provided. The model will not make meaningful predictions!")

        self.model.eval().to(self.device)

        # Initialize transform for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

        # Initialize face detector
        self.face_detector = MTCNN(keep_all=True, device=self.device)

    def detect_face(self, image):
        """Detect faces with debug information"""
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            boxes, _ = self.face_detector.detect(image_rgb)

            if boxes is None:
                print("No faces detected in the image")
                return None

            print(f"Detected {len(boxes)} faces")
            return boxes

        except Exception as e:
            print(f"Error in face detection: {str(e)}")
            return None

    def preprocess_face(self, face):
        """Preprocess face with debug information"""
        try:
            if face.shape[0] == 0 or face.shape[1] == 0:
                print("Invalid face crop dimensions")
                return None

            face_pil = Image.fromarray(face)
            face_tensor = self.transform(face_pil)
            return face_tensor.unsqueeze(0)

        except Exception as e:
            print(f"Error in face preprocessing: {str(e)}")
            return None

    def check_liveness(self, image):
        """Check liveness with detailed debugging"""
        faces = self.detect_face(image)

        if faces is None or len(faces) == 0:
            print("No faces detected for liveness check")
            return 0.0

        results = []
        for i, box in enumerate(faces):
            try:
                x1, y1, x2, y2 = map(int, box)

                # Ensure valid coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

                # Extract and preprocess face
                face = image[y1:y2, x1:x2]

                if face.size == 0:
                    print(f"Invalid face crop for face {i+1}")
                    continue

                face_tensor = self.preprocess_face(face)
                if face_tensor is None:
                    continue

                # Get prediction
                with torch.no_grad():
                    output = self.model(face_tensor.to(self.device))
                    probabilities = torch.softmax(output, dim=1)
                    score = probabilities[0][1].item()
                    print(f"Face {i+1} liveness score: {score:.3f}")
                    results.append(score)

                # Draw rectangle and score
                color = (0, int(255 * score), 0)  # Color based on score
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f'Score: {score:.3f}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            except Exception as e:
                print(f"Error processing face {i+1}: {str(e)}")
                continue

        if not results:
            print("No valid liveness scores computed")
            return 0.0

        mean_score = np.mean(results)
        print(f"Mean liveness score: {mean_score:.3f}")
        return mean_score

    def analyze_depth_and_movement(self, video_frames):
        depth_model = DepthAnalysisModel()
        depth_scores = []
        for frame in video_frames:
            # Preprocess the frame for depth analysis
            frame_resized = cv2.resize(frame, (64, 64))  # Resize frame to match depth model input
            frame_tensor = transforms.ToTensor()(frame_resized).unsqueeze(0)
            depth_output = depth_model(frame_tensor)
            depth_scores.append(depth_output.item())

        # Check for subtle movements by analyzing differences between frames
        movement_score = 0
        for i in range(1, len(video_frames)):
            diff = np.sum(np.abs(video_frames[i] - video_frames[i-1]))
            movement_score += diff

        return np.mean(depth_scores), movement_score / len(video_frames)

def run_demo(model_path=None):
    print("Initializing detector...")
    detector = LivenessDetector(model_path)

    print("\nTaking a photo...")
    frame = colab_webcam()  # Capture a single frame from the webcam
    frames = [frame]  # Store the captured frame in a list for analysis

    print(f"\nCaptured 1 frame for analysis.")
    print("Analyzing liveness...")

    score = detector.check_liveness(frames[0])  # Check liveness on the captured frame
    print(f"\nFinal liveness score: {score:.3f}")

    # Run depth and movement analysis
    average_depth, movement_score = detector.analyze_depth_and_movement(frames)
    print("Average Depth Score:", average_depth)
    print("Movement Score:", movement_score)

    # Display the result
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Usage example:
run_demo()





resnet.eval()


