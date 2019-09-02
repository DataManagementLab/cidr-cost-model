# DBMS Fitting: Why should we learn what we already know?
Proof of concepts for our vision paper

![Cost Estimates](/ml/plots/3d_attributes_tablesize/3d_attributes_tablesize_tuple_widths=16_selectivity=0.0_est.pdf)

## Abstract 
Deep Neural Networks (DNNs) have successfully been used to replace classical DBMS components such as indexes or query optimizers with learned counterparts.
However, commercial vendors are still hesitating to put DNNs into their DBMS stack since these models not only lack explainability but also have other significant downsides such as the requirement for high amounts of training data resulting from the need to learn all behavior from data.

In this paper, we propose an alternative approach to learn DBMS components.
Instead of relying on DNNs, we propose to leverage the idea of differentiable programming to fit DBMS components instead of learning their behavior from scratch.
Differentiable programming is a recent shift in machine learning away from the direction taken by DNNs towards simpler models that take advantage of the problem structure.
In a case study we analyze and discuss how to fit a model to estimate the cost of a query plan and present initial experimental results that show the potential of our approach.