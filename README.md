# Introduction
In recent years, technological advances in the field of face recognition and deep learning
algorithms have allowed for new applications and implementations in industry and academia.
Hereby, the connection between face recognition and deep learning algorithms presents to be
appealing to all kinds of businesses. One famous example is Facebook’s tool “Deepface”, which
allows the automatic tagging of faces in user generated content. Particularly in the security and
marketing industries, face recognition is achieving great attention.
In this paper, experimental work with the face recognition library Dlib is presented,
where the authors’ aim was to detect, identify and model celebrity faces in historic films.
The overall aim of our efforts was to develop a model, that would be able to accurately identify
four figures (namely Brigitte Bardot, Charles De Gaulle, Elisabeth Reine Mere and Jean Paul
Belmondo) in a series of short news videos. We will compare three different models for our
approach: direct face comparison using Euclidean distance, K-Nearest-Neighbors, and neural
networks. Our results show neural networks are the best option. For each, we will discuss data
preprocessing, the advantages and limitations, and potential future work.
# Technologies
## Dlib and face_recognition
Dlib is a popular and open source face recognition library written in C++. “Its design is
heavily influenced by ideas from design by contract and component-based software engineering”
(Wikipedia Dlib, 2018), meaning that it consists of many and sometimes independent software
components. The development of Dlib began in 2002 and has since expanded into many areas.
Since 2006, Dlib also contains machine learning and linear algebra components. In the project,
we choose to work with the Python implementation of Dlib and a Python package called
face_recognition. This was a purposeful choice as most data science tools are written in Python
and we wanted to facilitate our implementations. face_recognition is a state-of-the art tool based
on Dlib; it uses deep learning and can be easily applied. In particular, we were interested in the
face encoding that the tool returns.
In order to construct a model that successfully identifies a face, the face encodings of
each image needs to be retrieved. This we did with the face_encoding function provided by the
face_recognition library. The face_encoding is returned as a vector of length 128, containing the
value of each of the 128 features.
## Face Recognition
### Detection
In general, face recognition involves two steps: detection and identification. In the
detection phase, the task is to process an image and determine whether it contains a face or not.
Using the face_recognition library and a function called “face_locations”, we detect the region
containing a face. If an image contains several faces, several faces will be detected.
The first step is to find facial landmarks. Facial landmarks are the key features of a human face,
and the face_recognition library uses the mouth, right eyebrow, left eyebrow, nose bridge, nose
tip, left eye, right eye, and upper and lower lip as facial landmarks. In fact, its face_landmarks
function has already been trained to understand the organization of pixels in an image that
represent a human face and its landmarks.
If all landmarks have been successfully identified, face_locations will return a vector
containing the positions of each face in the image. We then can draw a red square around the
face to make the detection visible to the human, or contour the facial landmarks as depicted below. 
Although the forehead is not outlined, we can assess that both faces were successfully detected.
<img src = "https://raw.githubusercontent.com/Horbaje/Spring18_Project/master/Regular_Landmark.PNG">
### Identification
Identification is an area of research and a rather difficult and complex task as the fundus
of people needing to be identified constantly changes. Additionally, technicians are faced with a
giant load of data that needs to be collected and processed in order to accurately identify people
in images and video. Complexity issues arise as the number of people to be identified increases.
If you have an image of an unknown person and there are only a few people in the set of known
people, it is easy to run through the list one by one to compare. However, imagine Facebook’s
giant user base and the amount of images and videos uploaded every day, all of which require
facial detection and recognition for tagging purposes. In this case, the number of users means
there needs to be a more efficient way of recognizing faces than direct comparisons.
Overall, most of our project work focused on the identification problem. We applied
three approaches: direct comparison, k-nn classification and neural networks.

## Detection Implementation
While experimenting with the dlib and face_recognition library, we wanted to ensure that
we know its strengths and weaknesses in detecting human faces. Therefore, we ran several test
images on the face_landmark predictor to determine its limitations.
### Results
From our results, we have identified four main aspects to a facial image that can help
determine if it can be recognized: the size of the face within the image needs to be large, the
image quality needs to be high (resolution and/or image size), the face needs to be turned
towards the camera, and there cannot be any outside objects blocking the view of the face. Note
this is not just limited to unrelated objects (e.g. a hand covering the mouth), but also include
things that are typically seen on the face such as sunglasses. Overall, we find that the human is
still superior in detecting faces compared to the machine.
In order for the face detection to work, all four of these qualities need to be present. If the
face itself is too small within the image, then even if the quality of the image is superior (high
resolution) and the face is facing the camera, the face detector will fail to detect it. In general,
there is a certain threshold in which a face smaller than a certain amount of pixels will not be
detected at all. Further, while regular glasses mostly do not influence the detection negatively,
sunglasses create great distortion. This can be seen in Image 2. The sunglasses confuse the
predictor, leading to an unsatisfactory result. Finally, faces that are shown from the side or from
an angle that may cover or distort facial landmarks may also not always be detected.

Image 2
<img src = "https://raw.githubusercontent.com/Horbaje/Spring18_Project/master/Sunglasses_Landmark_half_features.PNG">

In general, the face_recognition library is very robust towards shadows and unusual lighting. As
a reference, the face_recognition library achieves an accuracy on 99.38% on the Labeled Faces
in the Wild benchmark.

# Experimental Techniques and Methods
## Data Collection and Preprocessing
We had two main sources for our data. For once, we got images from Google Images.
Secondly, we were provided with the historic videos by the INA. To begin with, our experiment
needed some data preprocessing. We began with our videos, which we needed to splice into
images by using cv2. For each celebrity, we had two videos that included that person. Because
our ultimate goal was to identify faces within these videos, we decided to use video images to
compose our test set. In order to quantify our results in an efficient way, we simplified things by
taking 30 screenshots of each person. This way we assess the accuracy of each model. Then, to
create our training set, we simply used Google Image Search. We carefully choose two images
per celebrity. It is important to note here that we had prior knowledge of our test set because we
knew we would be working with these old historic films. For example, we knew that the people
depicted in our films would be in black and white, so we specifically selected black and white
images for our training model. Additionally, we selected images with a high quality, the person
in the image faced the camera and the faces in the images were large.
## Direct Comparison
In our first experiment, we used face_recognition’s facial encodings feature. When
considering this in a 128 dimensional space, we can say that the closer two vectors are to each
other in space, the more likely it is that the vectors represent two images of the same person. For
the direct comparison method, we compared each test face individually with our training set of
10 images.
