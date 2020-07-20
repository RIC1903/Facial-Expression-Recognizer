# Facial Expression Recognition
Detect faces and recognize the expression of the faces in your image/video. We use Tensorflow and Keras to build the Facial Expression Recognition model. 
The model has four convolution layers and two fully-connected layers after flattening. The optimizer used is Adam optimizer. The model obtained an accuracy of 66.72%. 

![CNN model](https://github.com/RIC1903/Facial-Expression-Recognizer/blob/master/model.png)

### Dependencies
- TensorFlow 2.x
- Keras
- OpenCV
- Flask
- Seaborn


Run the code below to install the dependencies required.
```python
pip install -r requirements.txt
```

### Usage
1. Extract the Training_Images.rar(contains folder train and test) to the cloned folder.
The training images will be in "./train" and testing images in "./test".

2. The Facial_Expression_Training.ipynb is used to train the model. It will dump model_weights.h5 and model.json after training.

3. The image processing part is done in camera.py ('haarcascade_frontalface_default.xml' is included in the project directory).
   - To capture video from webcam:
     ```python
     def __init__(self):
          self.video = cv2.VideoCapture(0)
     ```
   - To capture video from a file:
      ```python
     def __init__(self):
          self.video = cv2.VideoCapture('sample_video.mp4')
     ```
4. The model.py gets the input frames from camera.py and returns the predicted expression from the predict_emotion function.
    ```python
    def predict_emotion(self, img):
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
    ```
    
    The emotions list --->
    ```python
    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]
    ```
    
5. The flask server runs on your localhost on port 5000 (http://127.0.0.1:5000/).
   - To see index.html of templates folder:
     ```python
     @app.route('/')
     def index():
      return render_template('index.html')
     ```
   - To directly see the video feed:
     ```python
      @app.route('/video_feed')
      def video_feed():
          return Response(gen(VideoCamera()),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
     ```
   



