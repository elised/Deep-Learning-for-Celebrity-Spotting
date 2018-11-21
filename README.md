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
