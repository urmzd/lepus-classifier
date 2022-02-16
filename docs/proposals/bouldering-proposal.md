# Classification of Bouldering Climbing Grade using Neural Networks

## Motivation for Problem Proposal

In rockclimbing, the difficulty of climbing routes is classified using a grade given by climbers, generally depending on the length or style of route, the challenge of the required moves, and how dangerous climbing conditions are. As a result, climbing grades are very subjective as they depend on a collective human assessment and the many existing grade scales are rather ambiguous when it comes to distinguihsing between closely related difficulty classes. This presents a challenging task in attempting to accurately predict the assignment of climbing grades using machine learning, with the potential to benefit the performance of climbers (particularly when training) and to aid in improving the ambiguity of standardized scales. 

With this project, we would explore predicting the climbing grade of Bouldering problems, climbing routes that are for free climbing without ropes or harnesses that are generally no higher than 6 metres, using the Hueco scale (V0 - V17).

## Data 
To explore this challenging task, we would use climbing route data collected from https://www.moonboard.com/. Moonboards are a standardized climbing wall made up of 142 rock holds on a 18x11 grid that are used for indoor bouldering training. Moonboard climbers utilize an app to load a problem route to the board, the sequence of holds marked by illuminated LEDs, so a large dataset of problems has been created by Moonboard and community users (over 30k problems were scraped by Duh & Chang in their 2021 work using RNN models for classication and route generation). 

## Related Works

- Dobles et al. (2017) employed and evaluated Naives Bayes, softmax regression, and Convolutional Neural Network classifiers to attempt to determine the difficulty grade of climbing routes, specifically using a dataset collected from Moonboard.com to standardize the data. They yielded the following results for each classifier resepctively: 34.0% 36.5% 34.0%
- Duh & Chang (2021) employed RNN architectures to explore classifying Moonboard climbing route grades and to generate new Moonbaord routes. Their 'GradeNet' architecture achieved 46.7% accuracy upon testing.
- Tai et al. (2020) applied Graph Convolutional Neural Networks (GCN) architectures previously used in NLP applications to classifying the climbing route grade of Moonboard problem sets, with their top model achieving an average AUC score of 0.73 across all classes.

Dobles, A., Sarmiento, J. C., & Satterthwaite, P. (2017). Machine learning methods for climbing route classification. Web link: https://cs229.stanford.edu/proj2017/final-reports/5232206.pdf

Duh, Y. S., & Chang, R. (2021). Recurrent neural network for moonboard climbing route classification and generation. arXiv preprint arXiv:2102.01788.
https://arxiv.org/pdf/2102.01788.pdf 

Tai, C. H., Wu, A., & Hinojosa, R. (2020). Graph neural networks in classifying rock climbing difficulties. Technical report. Web link: http://cs230.stanford.edu/projects_winter_2020/reports/32175834.pdf

Kempen, L. (2018). A fair grade: assessing difficulty of climbing routes through machine learning. Formal methods and tools, University of Twente. https://fmt.ewi.utwente.nl/media/30-TScIT_paper_46.pdf 

Scarff, D. (2020). Estimation of Climbing Route Difficulty using Whole-History Rating. arXiv preprint arXiv:2001.05388. https://arxiv.org/pdf/2001.05388.pdf

## Our Planned Approach

##
