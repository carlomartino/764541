# ML_Group_09

**MACHINE LEARNING (TITLE)**

***Predicting the Popularity Score of Music Tracks***

* Alexandra Hassan - 764061 - alexandra.hassan@studenti.luiss.it
* Carlo Martino - 764541 - carlo.martino@studenti.luiss.it
* Simone De Benedictis - 764941 - simone.debenedictis@studenti.luiss.it

**1) Introduction**

Every now and then a new, different song becomes a hit: we hear it everyday on each radio station, we may hum the words even if we’ve never heard it in full. Sometimes the song is a ballad, sometimes it is a powerhouse disco hit.
That poses a question: which are the features that make a song so popular? Is it possible to predict how popular a song will be based on qualities like its energy, acoustic-ness or danceability?
This is what we will analyze and focus on in our project, thanks to data provided by the Music Intelligence Department (MID), which analyzes a library of songs that we can gain insight on to potentially give recommendations for future releases.

In the data set, to look at the characteristics that may make a song stick in your head, we were provided with 20 variables: 
* A unique Track ID for each song
* The names of the Track, Artist and Album
* The Genre of the Track
* Variables related to the structure of the Track, like Duration, Key, Mode, Time Signature and Loudness
* Variables related to the production of the Track, like Danceability, Energy, Acoustic-ness, Instrumental-ness and Tempo
* Variables related to the content of the Track, like Speechiness, Liveness, Valence and whether it contains Explicit lyrics
* The Popularity of the track, our target, variable, with a value from 0 to 100

**2) EDA & Clustering**

As we looked at the data set, we noticed a few things that we thought should be fixed.
Therefore, when we started our Exploratory Data Analysis, or EDA, we made some choices regarding the data cleaning, in order to have a cleaned data frame that would be good for the future models.
First, we noticed that certain songs appeared in the data more than once, either with the same track ID but different track genres, or with a different track ID altogether. To solve this, we decided to create a new variable that merged the Artist name and the Track name into one, which we named Artist Song: based on this new variable, we dropped all duplicates, then moved the row to the left of the dataframe, as it represented our new beginning.
Unfortunately, we noticed that this did not solve our issues completely: certain songs, like the ones included in soundtrack albums, were still featured twice as they appeared with two different names, one with the formula ‘ARTIST - SONG’ and one with the formula ‘ARTIST - SONG (from MOVIE/TV SHOW)’. Since in music streaming services, the overall listens of the two versions are counted together, there was no need to have both: as a matter of fact, we noticed that one of the two versions almost always had popularity 0. 
Because of this, we decided to drop all songs with popularity 0, since we wanted to avoid all confusion with the remaining duplicates, and also felt that songs with such popularity may not be relevant enough for our analysis overall.

Since we felt that the Artist may be a good variable to consider, we wanted to see how many unique artists were featured in the data set to potentially One Hot Encode it, but the result we got was over 30.000. In order to avoid a model that was too computationally expensive, we decided that we would therefore drop the column and not consider it for the Regression.
Similarly as the Artists, we also decided to drop the Album name, the Track name and the Track ID, as well as all the null values in the data.

Looking at the different variables, we realized that there were three specific ones that needed Encoding:
the Explicit variable had a value of either True or False, so we decided to Label encode it, giving it a value of either 0 or 1
the Track Genre variable had over 100 possible values, so we decided to One Hot encode it since, even if it may make the model more computationally expensive, we thought it was important to consider the genre when predicting the popularity
the Key variable had a value in the range from 0 to 11, as it indicates one of the 12 major scales/keys that a song can be in: we decided to One Hot encode this one as well.
With this new encoded data frame, we once again dropped all null values: the result was a data set of 53.451 rows and 142 columns, which we used to tackle the issue of finding the popularity score.

Continuing our EDA, we selected the features for which we wanted to examine the distribution: in this group, we excluded “Track Genre” and “Key” but included Explicit.
We created Histograms and Box Plots, which showed us that the variables had different distributions and did not necessarily follow similar paths: danceability, for example, had a more or less normal distribution while, for energy, the number of songs increased as energy itself increased.
The last step of our EDA was investigating the relationship among the different features using a Correlation Matrix.
Through this, we were able to see a few interesting connections, like the negative correlation between energy and acousticness, or the positive correlation between the former and loudness; nevertheless, when looking at the popularity, which was our target, it was clear that none of the audio features had a big impact on it on their own, with the highest correlation being with danceability (0.08).

As we finished our EDA, the following step was Clustering.
We did Clustering with a very specific aim in mind: checking if it was possible to cluster songs based on their audio features and having those clusters match with the track genre of those songs.
Utilizing K-Means, we created over 100 clusters, and then compared the labels of the clusters with the label given by the track genre.
In order to objectively see whether this was a good way technique, we calculated the Silhouette Score and the Homogeneity Score: the value of both metrics ranges from 0 to 1, and they measure the quality of the clusters by taking into consideration, respectively, the cohesion and separation between clusters and the degree to which clusters contain only samples of a single label; in our case, the values were 0.10 and 0.26 respectively, which were both low and showed us that this was not a good way to tackle the issue and proceed.

**3) Regression & Classification**

Since the ultimate goal of the analysis was to predict the Popularity score, which is a continuous variable with values ranging from 0 to 100, we decided to tackle this as a Regression problem. As we will see later, however, we also applied a classification model.

Starting with Regression, we split the data into Train and Test data, defining the popularity as the Y and utilizing the data frame with the encoded variables as our X, after removing a few additional columns from it.
We defined that 25% of the data should be used as the Test set, while the remaining 75% for the Training set; finally, we scaled the data using Standard Scaler / MinMax Scaler, and proceeded.

We tried multiple models, including Support Vector Regression, Gradient Boosting and Decision Tree. Our top 3 models, however, ended up being:
Linear Regression: the most simple and widely used algorithm, it tries to find the best-fit line that minimizes the sum of the squared differences between the predicted and actual values. While this model may not capture complex non-linear relationships, we decided to still use it as a baseline and include it.
Random Forest: an ensemble learning algorithm that combines multiple decision trees to make predictions, where the final prediction is made by aggregating the predictions of all the trees.
eXtreme Gradient Boosting (or XGBoost): a gradient boosting algorithm, it creates an ensemble of weak prediction models and sequentially improves them by minimizing a loss function.
For each of this, we calculated the R-squared, which is a statistical measure that determines the goodness of fit.
This was done to evaluate the quality of each model and determine how much each model could capture the variation in the target variable, which is the popularity score.
The value of the R-squared ranges from 0 to 1, and the ideal situation is when the score is maximized and close to 1, which indicates a perfect fit: in our case, the values of the R-squared were 0.44 for the Linear Regression, 0.50 for the Random Forest and 0.51 for XGBoost.
Since the latter showed the most promise, we decided to launch a grid search using the XGBoost, but we were only able to increase the R-squared to 0.527.

As the results of the R-squared did not feel completely satisfactory, we realized that we had an opportunity: turning the the problem at hand into an issue of Classification by trying to find a class of popularity, as opposed to a specific score, potentially increasing the quality of the final result and increasing the interpretability of the results.
We decided to apply both a Random Forest classifier and XGBoost classifier, divide the popularity in 4 possible classes (1-25, 26-50, 51-75 and 76-100), and once again divide the Train and Test sets before scaling.
In order to evaluate the two models, we used three metrics:
Accuracy, which measures the overall correctness of a model by calculating the percentage of True labels (Negative and Positives) over the total amount of observations
Precision, which focuses on the model's ability to identify True Positives and minimize False Positives errors.
The F1 Score, which combines Precision and Recall, another metric that helps minimize False Negatives, to give an overall balanced evaluation.

**4) Results**

In order to see the results and be able to interpret them and derive our conclusions, we created a final data frame of the songs contained in the Test set, attaching to the audio features the predicted Popularity Score based on the Regression model and the predicted Popularity Class based on the Classification.
We decided to consider the top 20 songs based on the Regression predictions and analyze the audio features of those to see whether or not there were similarities that could show us an ideal profile of a future hit song.
Based on our results, we noticed that popular songs tend to have considerable danceability and energy, be of the XX genre, and…

**5) Conclusions**

This analysis was an interesting way to see whether it is possible to measure the success of the song on objective, mathematical features provided by the MID. 
Through our results, it is possible to conclude that while it is, in part, possible to do so, there are some limitations because the subject at hand, in our opinion, includes elements that are beyond the scope of measurability.
Songs can become popular for multiple reasons that are not taken into consideration in the data set: social media, world events, live performances, a great music video or simply the fact that the artist that sings it has a very dedicated fan base that would make the song become a hit no matter its sound.
This is why the evaluation of the models used is not as high as we may have hoped, because songs can have the same popularity and have completely different audio features.
Does this mean that this analysis has no value? Absolutely not! It can be important for music labels to be aware of current trends and to have their artists release a single with a sound that feels fresh and current, if they’re looking for a song that audiences will likely enjoy and stream.
Ultimately, however, we think an artist will maintain success and stability if they’re true to themselves and appear as genuine to the audiences, and if the audience is able to capture their personality and essence through the song they’re listening to: because of this, music labels should not put too much pressure or give too many directions to artists that may limit their artistic creativity.
