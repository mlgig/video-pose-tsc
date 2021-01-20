<b>Abstract</b>. In this paper we present an approach for the classification and interpretation
of human motion from video data. Our work builds upon the state-of-the-art advances in
the area of Human Pose Estimation for video and Multivariate Time Series Classification
and Interpretation. Our goal is to facilitate physiotherapists, coaches and rehabilitation
patients by providing feedback after the execution of physical exercises. Recent work in
sports science focuses on data collection with sensor devices, followed by a feedback
step to the user. For example, the participant executes an exercise, and an application
tells them whether the exercise was executed correctly or not, and what part of the
movement was not executed correctly. Using sensors for collecting motion data has its
challenges, for example, sensor devices require careful calibration, may not capture the
full richness of the movement and are not easily accepted by users. Instead, we work
with video data captured via mobile cameras, transform the video into time series via
human pose estimation, train time series classifiers, and deliver feedback to the user
through a time series classification and explanation step. We evaluate our approach
on a real-world Crossfit Workout Activities dataset collected by the Personal Sensing
Group at the Insight Centre for Data Analytics, University College Dublin, Ireland. We
show that, although data capture with video and pose estimation is noisy, we obtain
encouraging results with this approach and can provide useful feedback to the users
