# 1D-Convolutional-Neural-Network
A basic 1D CNN for predicting the Bitcoin Price Index

This project was intended to probe the dependence of the future bitcoin price index on its previous behavior. A 1d convolutional network was chosen under the assumption that, on a sufficiently small timescale, the time-dependent convolution function becomes approximately static.

Imports JSON objects containing price information, splits it into equally-sized training and reference sets. Each input element is a matrix containing 3 vectors of integer M bitcoin prices, derivatives, and second derivatives. The network takes a set of elements and transforms them to a single vector with N future bitcoin prices starting with the M+1 timestep with respect to the inputs.

This experiment was performed rather capriciously. The data preprocessing, in particular, was suboptimal, and I now suspect that a more intelligently-designed network would not require any derivatives as input. A more sophisticated approach is demonstrated in the MarketPrediction_Pytorch repository.
