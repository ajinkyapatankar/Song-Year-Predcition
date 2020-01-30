Song Year Prediction on Million Song Dataset
using Pyspark
Akshat Kumar,Ajinkya Patankar
ak1648@rutgers.edu
aap256@rutgers.edu
December 18, 2018
Abstract
Timbre is the quality of a musical note, sound, or tone that distin-
guishes dierent types of sound production, such as voices and musical
instruments This project focuses upon prediction of the release year of a
particular song by using a song's inherent timbre features. It will serve as
a useful topic because people tend to have some anity to dierent genres
of music throughout their life. This research will also help in determining
the long term evolution of music over the years and could be a useful basis
for recommending songs. The study could further be extended to gener-
ate new songs by learning the audio features of songs over the years. In
this research, we have applied four dierent predictive models on Million
Song Dataset and these are built on top of Apache Spark. The models
have been compared based on their accuracy to predict the release year as
well as their performance on a single node and multi-node Apache Spark
cluster environment.
1 Introduction
1.1 Problem Description
In simple terms , timbre is a unique feature of any sound vibration, other than
frequency, amplitude and modulation, which makes it dierent from every other
sound. Two sounds can have same frequency and amplitude but certainly not
the same timbre. [4]
Example, The sounds played by a violin and a piano at the same note sound
dierent because of timber.
The Million Song Dataset contains 90 timbre features.So, the pertinent question
with regard to this data was:
Does there exist a strong link between the audio features of a particular era to
the types of songs created then?
1
So, essentially what we are trying to do is take the timbre features of the songs
and trying to establish a link with the release year of these songs.
1.2 Problem Signicance and Goal
People tend to have some anity to the genres of music(such as community
events, college and high school), thus predicting the year of any particular song
will serve as a useful topic.
Also, an accurate model of the variation in the audio features of songs through
the years could be useful in predicting in long terms, the evolution of various
music genres.
Prediction of the release years of songs would also be a useful basis for song
recommendations.[3]
2 Dataset
The Million Song Dataset, is a freely-available collection of audio features and
metadata for a million contemporary popular music tracks available at UCI
Machine Learning Repository. Attractive features of the Million Song Database
include the range of existing resources to which it is linked, and the fact that it
is the largest current research dataset in this eld. [1]
The MSD contains metadata and audio analysis for a million songs that were
legally available to The Echo Nest. The songs are representative of recent west-
ern commercial music. The MSD stands out as the largest currently available
for researchers.
2.1 The Timbre Features
Timbre, in literal sense means, Tone Quality that distinguishes dierent types
of sounds.
The rst feature is the decision label Year (target), ranging from 1922 to 2011
and then there are 90 attributes[5]
TimbreAverage[1-12]
TimbreCovariance[13-90]
Following is a table, that lists down some of the features of the songs :
Field Type Description
Analysis Sample Rate 
oat rate audio
danceability 
oat algorithmic estimate
energy 
oat energy from listener view
key condence 
oat condence measure
Table 1: Sample Timbre Features Description
2
2.2 Dataset Characterstics
The MSD contains audio features and metadata for a million contemporary
popular music tracks. It contains: 280 GB of data, 1, 000, 000 songs/les, 44,
745 unique artists, 7, 643 unique terms (Echo Nest tags), 2, 201, 916 asymmetric
similarity relationships and 515, 576 dated tracks starting from 1922
3 Approach and Methodology
3.1 Sample Data Analysis
We imported the data set and took rst two results of the same. Following was
the output.
3.2 Data Cleaning
The above cluttered results were represented in a presentable format, by cap-
suling the timber features with its corresponding label which is year in which
the song was released.
3.3 Timbre Feature Analysis
A sample of timbre features are selected from the data and their values are
being compared against each other to know whether they lie in the same scale
or they have dierent scales and require normalization.
t1,t2,t3,t4,t5 represent sample timbre features.
As you can see from the below graph, value of features vary in a large magnitude
and there we will do feature scaling to bring down every feature on a scale of 0
to 1.
3
We have also plotted the value of each of the dataset features for 50 randomly
selected song tracks to get a sense of the dierence in the feature values which
would be helpful for us in further analysis.
4
3.4 Rescaling and Normalization of Timber Features
After the timbre feature analysis, we know that feature scaling and normaliza-
tion has to be done before we could move forward.
The formula below is calculated by subtracting the maximum value of a partic-
ular feature from each of it's values and dividing the result with the dierence
in the maximum and minimum value of that feature.
Formula used for rescaling and normalization is:
scaledF eature[i] = feature[i]􀀀max
max􀀀min (1)
3.5 Finding Track Count per year
We found out the smallest as well as the largest value of the years, correspond-
ing to each of the songs in our data set. Minimum value was 1922, maximum
was 2011.
The number of songs was then plotted against the year in which it was re-
leased.After a little more renement of the corresponding labels, the training
data set was ready.
5
3.6 Shifting of Labels
In order to simplify the complexity and enhance the eciency of the predictive
model, all the values of the labels will be shifted, in order to start from 0.
This means that the Value of the rst label, ie 1922 will be shifted to the value
0 1923 :1 , 1924: 2 so on.
3.7 Feature addition
Till now, we have been using 12 timbre features, however accuracy of the system
enhances rapidly, if more number of relevant features are added. For that, we
used the concept of 2-way interaction among the features. Suppose there are
3 features a,b and c. Then apart from these three features, we can add some
more features like a*a, a*b, a*c, b*b, b*c etc.
6
3.8 Visualization of Normalized Data
Data has been visualized in the form of a heat-map. This heat map clearly
illustrates the features whose value is approaching 1 after rescaling and normal-
ization.
line-graph has also been generated, which represents the number of songs per
year, for all the years in the range 1922-2011.
7
3.9 Dataset Division for Model Development
We have also created a pie-chart, that represents the breakdown of the amount
of data we would be using in the future for Training/Validation/Testing sets.
8
4 Model Framework Implementation
The Model Development Stage is the next stage in our project. We have built
four machine learning models namely Baseline Model, Linear Regression
Model, Random Forest Model and Gradient Boosted Trees model on
top of Apache Spark on the transformed data for song year prediction. In the
end, we will be using the comparison between Regression, Random Forest
and Gradient Boosted models as the yardstick of knowing which is better in
terms of both accuracy and computational time.[2]
4.1 Development of Baseline Model
A baseline model has been developed that predicts the average of all the years
in our dataset. In our case, the average comes out to be 76.4 (after shifting the
year label)
We calculated the Root Mean Square Error (RMSE) on training, test and
validation datasets and have found the RMSE for training dataset to be 10.93
approx, for testing dataset to be 10.96 approx and for validation dataset to be
10.897 approx.
4.2 Development of Linear Regression based model
A linear regression based model has been implemented to predict the year of
a particular song. Comparing the results so obtained with those of Baseline
Model, we can see that the Linear Regression Model has outperformed the
former at rst decimal place itself.
9
Here, we nd that the Root Mean Square Error for training dataset is 10.006
approx, while that of testing dataset is 10.019 approx.
4.3 Random Forest Implementation
4.3.1 Training model with dataset
Random forests train a set of decision trees separately, so the training can be
done in parallel. Instead of searching for the most important feature, it searches
for the best feature amongst random subset of features.
The algorithm injects randomness into the training process so that each
decision tree is a bit dierent. Combining the predictions from each tree reduces
the variance of the predictions, improving the performance on test data. The
time taken by Rondom Forest trainig process is 2194.14
10
4.3.2 Testing and Validation dataset application
4.4 Implementation of Gradient Boosted Trees
Gradient boosting iteratively trains a sequence of decision trees. On each it-
eration, the algorithm uses the current ensemble to predict the label of each
training instance and then compares the prediction with the true label. The
dataset is re-labeled to put more emphasis on training instances with poor pre-
dictions. Thus, in the next iteration, the decision tree will help correct for
previous mistakes.
11
4.4.1 Training model with Gradient Booster
Once, we realize that the model has reached a stage that residuals cannot be
remodelled further, or it has perfected the model, we can stop the iterations.
The specic mechanism for re-labeling instances is dened by a loss function.
With each iteration, GBTs further reduce this loss function on the training data.
12
5 Single vs Multi-node Performance Compari-
son
Single node V/S Multi-node Spark cluster environment:
We implemented the three models, namely Linear regression, random forest and
gradient boosted trees and tested them on a single node and multi-node Spark
environment.
5.1 Conguration Specication
Conguration of Single Node system :
The overall single node processing has been done on a i5 system , with 16 GB
RAM, exclusively allocated to the virtual machine. The processing is distributed
across 4 cores.
Conguration of Multi node system :
The computation is done with the help of 5 nodes. Out of which 1 is the master
node and 4 of them are slaves. Each of the four slaves have i5 processors with
4GB RAM , having 1 core each.
13
5.2 Performance Benchmarking
 Data Count - 1 node :
 Data Count - 5 nodes :
14
 Finding the minimum and maximum year - 1 node :
 Finding the minimum and maximum year - 5 nodes :
15
 Linear Regression Model - 1 node :
 Linear Regression Model - 5 nodes :
16
 Gradient Boosted Model - 1 node :
 Gradient Boosted Model - 5 nodes :
17
 Random Forest Model - 1 node :
 Random Forest Model - 5 node :
18
6 Results
For the purpose of implementation, Apache 2.3.0 single node cluster with a mas-
ter and worker set was used and for multinode, a set of 5 nodes i.e one master
and 4 workers, was used.
The above gure shows the time comparison of single node and multiple
nodes during training of predictive models. Given below is the diagrammed rep-
resentation of the number of records per year, after the segregation of datasets.
19
Following is the algorithmic view of the work done
20
Graphical Representation of Time comparison of computation
Following is the table for comparing the accuracy of the models through RMSE
MODEL RMSE
Baseline 10.93
Linear Regression 10.01
Random Forest 9.66
Gradient Boosted Trees 9.77
Table 2: Model RMSE Comparison
21
7 Conclusions
From results, it is clear that, distributed computing greatly reduces the com-
putation time of training our models. Thus, we highly recommend the use of
Apache Spark framework, for faster results. Apart from that, we have been also
able to establish which algorithm, can be best used for the prediction.
Random Forest performed the best in terms of accuracy calculated in terms of
RMSE. Linear Regression performed well both in single and multi-node cluster
with the least training time. On an average, multi-node cluster ran the the built
models 2.5X faster in comparison to a single node cluster.
By creating a system that predicts the release year of a particular song, we can
develop a more robust model for a song recommender system. Conventional
recommender systems, generally take into account just the user ratings, but
integrating year prediction, we can also include audio features.
8 Future Scope
In our proposed work, we used a sub-set of the million song dataset. In future,
we can scale up the system for all one million data samples using the Amazon
Web Services (AWS). We can also investigate the trend of time complexity by
increasing the number of nodes in the cluster step by step.
Furthermore, relationships can also be established between the instruments used
some time back and now, by further studying the timbre features of instruments
as well.
References
[1] https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd.
[2] https://cseweb.ucsd.edu/classes/wi17/cse258 a/reports/a028.pdf.
[3] https://vdocuments.mx/music-similarity-measures-whats-the use.html.
[4] https://www.ee.columbia.edu/ dpwe/pubs/BertEWL11 msd.pdf.
[5] https://www.kaggle.com/vinayshanbhag/predict-release-timeframe-from-
audio features/data.
22
