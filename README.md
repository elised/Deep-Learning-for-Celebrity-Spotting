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
<img src = "https://raw.githubusercontent.com/Horbaje/Spring18_Project/master/SP_Comparison.png">
This worked because our training set wasn’t too large. But this method is not scalable. Imagine
10,000 images of which we have to compare each one with 20 faces of our training set. This
would take quite long and even longer if we would add images subsequently. Another limitation
to this method is that there is no mechanism for handling cases where an encoding matches two
or more known people. We could build a function for this; for example, we could write in a stop
condition for when it reaches a match, but then that leaves open the possibility of unnecessary
bad matches. To solve this more elegantly, we could employed a k-nearest neighbors algorithm,
bringing us to our next testing method. The direct comparison model achieved an accuracy of
61.76%.
## k-Nearest Neighbors
K-nearest neighbors (k-nn) is the natural evolution of direct comparison, as we can
compare against the entire encoding space at once. K-nn is a classification algorithm, which
belongs to the family of supervised learning algorithms. “ Informally, this means that we are
given a labelled dataset consisting of training observations (x, y) and would like to capture the
relationship between x and y. More formally, our goal is to learn a function h:X→Y so that given
an unseen observation x, h(x) can confidently predict the corresponding output y (Zakka, 2016).”
Our labeled dataset is again the training dataset containing the labelled images from Google
Images. We have two images each celebrity plus the label corresponding to that class. The idea
is the following, we first loop through all celebrities in the training set and encode each one.
Then we determine how many k- nearest neighbours to use. We determined the number of k-nn
by taking the square root of the number of encodings. Generally, the number of k-nn is open to
experimentation and depends on the nature of the problem. We then trained the k-nn algorithm.
During training, we group the most similar encodings. Once the training is done, the trained
classifier is returned on which we made predictions. When we apply the trained classifier to our
test data, faces (encodings) are assigned to clusters. The assignment is done through a sort of
majority voting, taking the distance between points and hence determining similarity.
<img src = "https://raw.githubusercontent.com/Horbaje/Spring18_Project/master/SP_kNN.png">

In order to achieve a higher accuracy, here it makes sense to implement a threshold. The lower
the threshold (or distance amongst datapoints), the closer the encoding of a face has to be to a
group, otherwise it will be discarded as “not the same person”. For faces of unrecognized
persons, we returned the name 'unknown'. If an image does not contain any faces, an empty
result will be returned. Finally, we return our prediction, removing the classes that could not be
assigned. The K-Nearest Neighbors algorithm achieved an accuracy of 77.67%. For our
implementation, we used the sklearn neighbors library.
## Neural Networks
Our last approach is to use neural networks. Neural Networks are roughly modeled after
how a brain functions, using individual nodes as signal processors. Each neuron can be excited
through a sort of action potential. Much simplified, we feed some input into our NN architecture,
the NN breaks up all features and we receive as output our classification. Neural Networks are
much more flexible than either k-nearest neighbors or direct comparison, because we are able to
assess each feature separately and improve our model by running multiple iterations to minimize
the loss. We used TensorFlow in our implementation for simplicity and breadth of Python
support. We chose not to include any hidden layers, because while they would have been helpful
in drilling down to the encoding level, we didn’t find it necessary as our encodings had already
been extracted via the face_recognition package. The face_recognition package itself is based on
a deeplearning algorithm. Additionally, the number of input classes does not exceed four (one for
each celebrity), meaning that there will no more hidden layers be required in order to address our
problem sufficiently.
<img src = "https://raw.githubusercontent.com/Horbaje/Spring18_Project/master/SP_NN.png">
After deciding on the architecture of our Neural Network, we have to decide on which activation
function to use, what learning rate, batch size, optimizer and training epochs. For most
parameters, there is no clear recipe. Rather, we try different ones and assess their performance on
our problem. We decided on a learning rate of 0.1, which is a standard learning rate for these
type of problems. We set the training epochs to 100. The batch size of the algorithm is 128- 128
respectively for 128 pixels.
As activation function, we used the softmax function and for the optimization we used the Adam
optimizer. The softmax is a “ generalization of the logistic function that "squashes" a K-dimensional vector
of arbitrary real values to a K-dimensional vector of real values, where each entry is in the range
(0, 1), and all the entries add up to 1”(Wikipedia, 2018). The benefit of applying the softmax are
that all values range from 0 to 1, as it is a more efficient extension of Stochastic Gradient
Descent and that it facilitates working with probabilities. In conclusion, The NN was the best
model and achieved an accuracy of 90.20%.
<img src = "https://raw.githubusercontent.com/Horbaje/Spring18_Project/master/SP_NN2.png">
## Results
All in all, the deep learning approach is most successful when building a model for face
recognition. Followed by K-Nearest- Neighbor and then comparison. The accuracy rates for each
approach are summarized in the following tables:<p>
<b>Using Both Historic Film Data for Test and Training</b>
  
| Classifier/ Model | Accuracy |
| ----------------- | -------- |
| Comparison        |  61.76%  |
| K-NN              |  77.67%  |
| Neural Network    |  90.20%  |
<p>
<b>Using Google Images for Training Data and Historic Images as Test Data</b>
| Classifier/ Model | Accuracy |
| ----------------- | -------- |
| Comparison        |  32.4%   |
| K-NN              |  74.8%   |
| Neural Network    |  59.8%   |

### Recall Rate
Accuracy rates by themselves often do not give enough insight or can even be the wrong
measure of performance. Therefore, we also take a look at sensitivity. Sensitivity is the
proportion of actual positives that are correctly identified as such. It is calculated by dividing the
true positives by the true positives plus the false positives.<br>
We achieved a recall rate of 1 with the Google Images in direct comparison.

# Challenges and Further Research
The quality of historic and generally old film can be a challenge. Besides that they are in
black and white, the films tend to be blurry. Furthermore, people depicted in old films are filmed
from a rather long distance (in comparison to today’s capturing). This causes faces to be rather
small and hinders the face encoding to perform properly. Quantifying the later is a topic that
should be investigated. Another challenge that we encountered in our project work is that
people’s faces are often shown from the side or from an not straight-forward angle. This again,
will cause the face encoder to struggle. For future work, we suggest adding a greater pool of test
images. Also those, which show the depicted persons from several different angles.
Emotions and face expression as well as individuals wearing sunglasses and hats can cause
distortion. In some of our historic films, Elizabeth's face was covered by her hat many times.
We do have indications that this lowered our success rate of detecting a face. Further
investigation in this direction should be pursued in order to improve results.
Another topic that should be explored in this context is on how to optimize video frame
processing for faster computation. It takes extremely much processing power to built a model for
an potential increasing number of celebrities. Then, applying this model to test videos and
generating a visual output again requires much computation work. Training should be done with
multiple views of the face. Not just straight from the front. Experiment with adding “unknown”
classified images to the training set.
At this point, our model is good at identifying the people that it was trained for. However, it
would be nice to make it more generalized.
# Conclusion
Working with deep neural networks is interesting and it is surprising how well they work.
Especially if we compare them to other models. Neural Networks become even more powerful
when sequenced. Yet a task such as identifying celebrities in video is quite complex. As humans
we are relatively confident in our predictions, however our model is not very good at making
abstractions. It’s very specialized. This project helped us to get a deeper understanding not just
of neural networks, but also face recognition, its challenges and video processing.
